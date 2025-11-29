"""
Minimal Pi-Long inspired webcam demo that runs Pi3 on sliding windows of
frames and streams the fused point cloud plus camera trajectory to rerun.

Usage:
    python scripts/webcam_rerun_demo.py --config configs/base_config.yaml \
        --camera 0 --chunk_size 8 --overlap 4 --max_frames 400 --spawn_viewer

Notes:
- rerun is required: `pip install rerun-sdk`
- This keeps everything in memory (no temp dirs) and applies simple SIM(3)
  alignment between overlapping chunks to maintain a consistent map.
"""
import sys
sys.path.append("./")

import argparse
import math
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

try:
    import rerun as rr
except ImportError as exc:  # pragma: no cover - rerun is optional at import time
    raise ImportError("Please install rerun-sdk to use this demo (`pip install rerun-sdk`).") from exc

from loop_utils.config_utils import load_config
from loop_utils.sim3utils import apply_sim3_direct, weighted_align_point_maps
from pi3.models.pi3 import Pi3


def _frames_to_tensor(frames: List[np.ndarray], pixel_limit: int = 255_000) -> torch.Tensor:
    """Resize frames to a uniform size (matching pi_long) and stack into [N, 3, H, W]."""
    sources = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    first_img = sources[0]
    w_orig, h_orig = first_img.size
    scale = math.sqrt(pixel_limit / (w_orig * h_orig)) if w_orig * h_orig > 0 else 1
    w_target, h_target = w_orig * scale, h_orig * scale
    k, m = round(w_target / 14), round(h_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > w_target / h_target:
            k -= 1
        else:
            m -= 1
    target_w, target_h = max(1, k) * 14, max(1, m) * 14

    tensor_list = []
    for img in sources:
        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_tensor = torch.from_numpy(np.array(resized)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor_list.append(img_tensor)

    return torch.stack(tensor_list, dim=0)


def _transform_camera_poses(poses: np.ndarray, s: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply SIM(3) to C2W camera poses (matches save_camera_poses logic)."""
    s_mat = np.eye(4)
    s_mat[:3, :3] = s * r
    s_mat[:3, 3] = t

    out = []
    for c2w in poses:
        transformed = s_mat @ c2w
        transformed[:3, :3] /= s
        out.append(transformed)
    return np.stack(out, axis=0)


class WebcamPiLongDemo:
    def __init__(
        self,
        config_path: str,
        camera_id: int = 0,
        chunk_size: int = 8,
        overlap: int = 4,
        max_map_points: int = 300_000,
        sample_ratio: float = 0.02,
        spawn_viewer: bool = False,
    ) -> None:
        self.config = load_config(config_path)
        self.config["Model"]["chunk_size"] = chunk_size
        self.config["Model"]["overlap"] = overlap
        self.camera_id = camera_id
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = max(1, chunk_size - overlap)
        self.sample_ratio = sample_ratio
        self.max_map_points = max_map_points

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        torch.manual_seed(42)
        np.random.seed(42)

        self.model = Pi3().to(self.device).eval()
        weight_path = self.config["Weights"]["Pi3"]
        weights = load_file(weight_path)
        self.model.load_state_dict(weights, strict=False)

        self.prev_chunk = None
        self.map_points: List[np.ndarray] = []
        self.map_colors: List[np.ndarray] = []
        self.cam_centers: List[np.ndarray] = []
        self.current_point_count = 0
        self.frame_buffer: List[np.ndarray] = []
        self.frames_seen = 0
        self.last_chunk_start = -self.step

        rr.init("Pi-Long Webcam Demo", spawn=spawn_viewer)

    def _filter_points(
        self,
        points: np.ndarray,
        images: np.ndarray,
        confs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a filtered/sampled subset of points and colors."""
        flat_points = points.reshape(-1, 3).astype(np.float32)
        colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255.0).astype(np.uint8)
        conf_flat = confs.reshape(-1)
        conf_threshold = float(np.mean(conf_flat) * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"])
        mask = (conf_flat >= conf_threshold) & (conf_flat > 1e-5)

        valid_points = flat_points[mask]
        valid_colors = colors[mask]
        if len(valid_points) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        if 0 < self.sample_ratio < 1.0:
            num_samples = max(1, int(len(valid_points) * self.sample_ratio))
            idx = np.random.choice(len(valid_points), num_samples, replace=False)
            valid_points = valid_points[idx]
            valid_colors = valid_colors[idx]

        return valid_points, valid_colors

    def _log_rerun(self) -> None:
        """Push the current map and trajectory to rerun."""
        if self.map_points:
            stacked_pts = np.concatenate(self.map_points, axis=0)
            stacked_cls = np.concatenate(self.map_colors, axis=0)
            if len(stacked_pts) > self.max_map_points:
                tail = len(stacked_pts) - self.max_map_points
                stacked_pts = stacked_pts[tail:]
                stacked_cls = stacked_cls[tail:]
            rr.log("map/points", rr.Points3D(stacked_pts, colors=stacked_cls))

        if len(self.cam_centers) >= 2:
            traj = np.stack(self.cam_centers, axis=0)
            rr.log("trajectory", rr.LineStrips3D([traj]))
            rr.log("camera/pose", rr.Transform3D(translation=self.cam_centers[-1]))

    def _run_inference(self, frames: List[np.ndarray]) -> dict:
        images = _frames_to_tensor(frames).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    preds = self.model(images[None])
            else:
                preds = self.model(images[None].float())

        preds["images"] = images[None].float().cpu()
        preds["conf"] = torch.sigmoid(preds["conf"])

        out = {}
        for key, value in preds.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.detach().cpu().numpy().squeeze(0)
            else:
                out[key] = value
        return out

    def _align_chunk(self, prev_chunk: dict, curr_chunk: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate SIM(3) from curr -> prev using the overlapping window."""
        point_map1 = prev_chunk["points"][-self.overlap :]
        point_map2 = curr_chunk["points"][: self.overlap]
        conf1 = np.squeeze(prev_chunk["conf"][-self.overlap :])
        conf2 = np.squeeze(curr_chunk["conf"][: self.overlap])
        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
        s, r, t = weighted_align_point_maps(point_map1, conf1, point_map2, conf2, conf_threshold=conf_threshold, config=self.config)
        return s, r, t

    def _update_map(self, chunk: dict) -> None:
        points = chunk["points"]
        images = chunk["images"]
        confs = chunk["conf"]
        filtered_points, filtered_colors = self._filter_points(points, images, confs)
        if len(filtered_points) == 0:
            return
        self.map_points.append(filtered_points)
        self.map_colors.append(filtered_colors)
        self.current_point_count += len(filtered_points)
        while self.current_point_count > self.max_map_points * 2 and self.map_points:
            dropped_pts = self.map_points.pop(0)
            self.map_colors.pop(0)
            self.current_point_count -= len(dropped_pts)

    def _update_cameras(self, poses: np.ndarray) -> None:
        for pose in poses:
            self.cam_centers.append(pose[:3, 3])

    def process_chunk(self, frames: List[np.ndarray]) -> None:
        chunk_pred = self._run_inference(frames)

        if self.prev_chunk is None:
            aligned_points = chunk_pred["points"]
            aligned_poses = chunk_pred["camera_poses"]
        else:
            s, r, t = self._align_chunk(self.prev_chunk, chunk_pred)
            aligned_points = apply_sim3_direct(chunk_pred["points"], s, r, t)
            aligned_poses = _transform_camera_poses(chunk_pred["camera_poses"], s, r, t)
            chunk_pred["conf"] = chunk_pred["conf"]  # kept for the next overlap alignment
        chunk_pred["points"] = aligned_points
        chunk_pred["camera_poses"] = aligned_poses

        self._update_map(chunk_pred)
        self._update_cameras(aligned_poses)
        self.prev_chunk = {
            "points": aligned_points,
            "conf": chunk_pred["conf"],
            "camera_poses": aligned_poses,
        }
        self._log_rerun()

    def run(self, max_frames: int = 0) -> None:
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")

        print("Press Ctrl+C to stop the demo.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frames_seen += 1
                self.frame_buffer.append(frame)

                rr.set_time_sequence("frame_idx", self.frames_seen)
                rr.log("camera/image_raw", rr.Image(frame[:, :, ::-1]))  # BGR -> RGB for visualization

                if len(self.frame_buffer) >= self.chunk_size and (self.frames_seen - self.last_chunk_start) >= self.step:
                    frames_for_chunk = self.frame_buffer[-self.chunk_size :]
                    self.last_chunk_start = self.frames_seen
                    self.process_chunk(frames_for_chunk)

                if max_frames and self.frames_seen >= max_frames:
                    break
        finally:
            cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam-based Pi-Long demo with rerun visualization.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to Pi-Long config YAML.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device id.")
    parser.add_argument("--video_path", type=str, default="", help="Video file path.")
    parser.add_argument("--chunk_size", type=int, default=8, help="Number of frames per chunk.")
    parser.add_argument("--overlap", type=int, default=4, help="Overlap between consecutive chunks.")
    parser.add_argument("--max_frames", type=int, default=0, help="Stop after this many frames (0 = unlimited).")
    parser.add_argument("--sample_ratio", type=float, default=0.02, help="Random sampling ratio for logged point clouds.")
    parser.add_argument("--max_map_points", type=int, default=5_000_000, help="Upper bound on total points kept for logging.")
    parser.add_argument("--spawn_viewer", action="store_true", help="Spawn the rerun viewer window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overlap >= args.chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    demo = WebcamPiLongDemo(
        config_path=args.config,
        camera_id=args.camera if args.video_path == "" else args.video_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_map_points=args.max_map_points,
        sample_ratio=args.sample_ratio,
        spawn_viewer=args.spawn_viewer,
    )
    demo.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
