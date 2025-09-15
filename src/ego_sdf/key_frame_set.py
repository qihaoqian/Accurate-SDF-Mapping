from dataclasses import dataclass

import torch

from ego_sdf.frame import RGBDFrame
from ego_sdf.utils.config_abc import ConfigABC
from ego_sdf.utils.keyframe_util import multiple_max_set_coverage


@dataclass
class KeyFrameSetConfig(ConfigABC):
    insert_method: str = "insert_method"  # naive | intersection
    insert_interval: int = 50  # number of frames between key frames
    insert_ratio: float = 0.85
    frame_selection: str = "multiple_max_set_coverage"  # multiple_max_set_coverage | random
    selection_window_size: int = 8
    frame_weight: str = "uniform"


class KeyFrameSet:
    def __init__(self, cfg: KeyFrameSetConfig, max_num_voxels: int, device: str):
        self.cfg = cfg
        self.max_num_voxels = max_num_voxels
        self.device = device

        self.frames: list[RGBDFrame] = []
        self.valid_indices: list[torch.Tensor] = []
        self.sample_counts: list[int] = []

        self.kf_seen_voxel_indices: list[torch.Tensor] = []
        self.kf_seen_voxel_num: list[int] = []
        self.kf_unoptimized_voxels: torch.Tensor | None = None
        self.kf_all_voxels: torch.Tensor | None = None

    def add_key_frame(self, frame: RGBDFrame, seen_voxel_indices: torch.Tensor):
        """
        Adds a key frame to the set.
        Args:
            frame: RGBDFrame to be added.
            seen_voxel_indices: indices of voxels seen by the frame.
        Returns:
            bool: True if the frame is added as a key frame, False otherwise.
        """
        if self.is_key_frame(frame, seen_voxel_indices):
            self.add_frame(frame, seen_voxel_indices)
            return True
        return False

    def is_key_frame(self, frame: RGBDFrame, seen_voxel_indices: torch.Tensor):
        if len(self.frames) == 0:
            return True

        voxels_unique, counts = torch.unique(
            torch.cat([self.kf_seen_voxel_indices[-1], seen_voxel_indices], dim=0),
            return_counts=True,
            sorted=False,
            dim=0,
        )
        n_intersection = torch.sum(counts > 1).item()
        n_union = voxels_unique.shape[0]
        iou = n_intersection / n_union
        if iou < self.cfg.insert_ratio:
            return True
        return False

    def add_frame(self, frame: RGBDFrame, seen_voxel_indices: torch.Tensor):
        self.frames.append(frame)
        self.kf_seen_voxel_indices.append(seen_voxel_indices)
        self.kf_seen_voxel_num.append(seen_voxel_indices.shape[0])

        valid_idx = torch.nonzero(frame.valid_mask.view(-1))
        self.valid_indices.append(valid_idx)
        self.sample_counts.append(sum(self.sample_counts) // (len(self.sample_counts) + 2))

        if self.cfg.frame_selection == "multiple_max_set_coverage" and self.kf_unoptimized_voxels is not None:
            self.kf_unoptimized_voxels.index_fill_(0, seen_voxel_indices.long(), True)

    def select_key_frames(self) -> list[int]:
        if len(self.frames) <= self.cfg.selection_window_size:
            return list(range(len(self.frames)))

        if self.cfg.frame_selection == "random":
            selected_frame_indices = torch.randperm(len(self.frames))[: self.cfg.selection_window_size].tolist()
            return selected_frame_indices

        if self.cfg.frame_selection == "multiple_max_set_coverage":
            selected_frame_indices, self.kf_unoptimized_voxels, self.kf_all_voxels = multiple_max_set_coverage(
                self.kf_seen_voxel_num,
                self.kf_seen_voxel_indices,
                self.kf_unoptimized_voxels,
                self.kf_all_voxels,
                self.cfg.selection_window_size,
                num_voxels=self.max_num_voxels,
                device=self.device,
            )
            return selected_frame_indices

        raise ValueError(f"Unknown frame selection method: {self.cfg.frame_selection}")

    def sample_rays(
        self,
        num_samples: int,
        key_frame_indices: list,
        current_frame: RGBDFrame | None,
        get_rgb: bool = False,
        get_depth: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # distribute num_samples to each frame based on the sample counts
        # higher sample count -> less samples

        frames: list[RGBDFrame] = [self.frames[i] for i in key_frame_indices]
        sample_counts = [self.sample_counts[i] for i in key_frame_indices]
        if current_frame is not None and current_frame != self.frames[-1]:
            frames.append(current_frame)
            sample_counts.append(sum(sample_counts) // (len(sample_counts) + 2))

        total_count = sum(sample_counts)
        n_frames = len(frames)

        if n_frames == 0 or num_samples == 0:
            return None, None, None, None

        if self.cfg.frame_weight == "uniform":
            samples_per_frame = [num_samples // n_frames] * n_frames
            for i in range(num_samples % n_frames):
                samples_per_frame[i] += 1
        else:
            if total_count == 0:
                samples_per_frame = [num_samples // n_frames] * n_frames
                for i in range(num_samples % n_frames):
                    samples_per_frame[i] += 1
            elif n_frames == 1:
                samples_per_frame = [num_samples]
            else:
                m = total_count * (n_frames - 1)
                samples_per_frame = [max(1, int(num_samples * (total_count - count) / m)) for count in sample_counts]
                # adjust to make sum exactly num_samples
                diff = num_samples - sum(samples_per_frame)
                for i in range(abs(diff)):
                    idx = i % len(self.frames)
                    if diff > 0:
                        samples_per_frame[idx] += 1
                    elif samples_per_frame[idx] > 1:
                        samples_per_frame[idx] -= 1

        rays_o_all = []
        rays_d_all = []
        rgb_samples_all = []
        depth_samples_all = []
        for frame_idx, frame in enumerate(frames):
            n_frame_samples = samples_per_frame[frame_idx]

            if frame_idx < len(key_frame_indices):
                i = key_frame_indices[frame_idx]
                self.sample_counts[i] += n_frame_samples
                valid_idx = self.valid_indices[i]
            else:
                valid_idx = torch.nonzero(frame.valid_mask.view(-1))
            sample_idx = valid_idx[torch.randint(0, valid_idx.shape[0], (n_frame_samples,))]
            sample_idx = sample_idx.view(-1)

            pose = frame.get_ref_pose()
            rotation = pose[:3, :3]
            sampled_rays_d = frame.rays_d.view(-1, 3)[sample_idx]  # (n_frame_samples, 3)
            sampled_rays_d = sampled_rays_d @ rotation.T  # (n_frame_samples, 3)
            sampled_rays_o = pose[:3, 3].view(1, 3).expand_as(sampled_rays_d)  # (n_frame_samples, 3)
            rays_o_all.append(sampled_rays_o)
            rays_d_all.append(sampled_rays_d)

            if get_rgb:
                sampled_rgb = frame.rgb.view(-1, 3)[sample_idx]  # (n_frame_samples, 3)
                rgb_samples_all.append(sampled_rgb)
            if get_depth:
                sampled_depth = frame.depth.view(-1)[sample_idx]  # (n_frame_samples,)
                depth_samples_all.append(sampled_depth)

        rays_o_all = torch.cat(rays_o_all, dim=0)  # (num_samples, 3)
        rays_d_all = torch.cat(rays_d_all, dim=0)  # (num_samples, 3)
        rgb_samples_all = torch.cat(rgb_samples_all, dim=0) if get_rgb else None
        depth_samples_all = torch.cat(depth_samples_all, dim=0) if get_depth else None
        return rays_o_all, rays_d_all, rgb_samples_all, depth_samples_all
