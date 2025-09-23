import argparse
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Optional

from tqdm import tqdm

from grad_sdf import torch
from grad_sdf.frame import Frame
from grad_sdf.gui_base import GuiBase, GuiBaseConfig, GuiControlPacket, GuiDataPacket
from grad_sdf.trainer import Trainer, TrainerConfig


class GuiTrainer:
    def __init__(self, gui_cfg: GuiBaseConfig, trainer_cfg: TrainerConfig):
        self.gui_cfg = gui_cfg
        self.trainer_cfg = trainer_cfg
        self.gui_cfg.scene_bound_min = self.trainer_cfg.model.residual_net_cfg.bound_min
        self.gui_cfg.scene_bound_max = self.trainer_cfg.model.residual_net_cfg.bound_max

        self.queue_from_gui = mp.Queue()
        self.queue_to_gui = mp.Queue()
        self.gui_process = mp.Process(target=GuiBase.run, args=(self.gui_cfg, self.queue_to_gui, self.queue_from_gui))

        self.trainer = Trainer(self.trainer_cfg)
        self.trainer.training_iteration_end_callback = self.training_iteration_end_callback
        self.trainer.training_frame_start_callback = self.training_frame_start_callback
        self.trainer.training_end_callback = self.training_end_callback

        self.control_packet: GuiControlPacket = GuiControlPacket()
        self.last_octree_time = 0.0
        self.last_octree_min_size = None
        self.last_sdf_slice_time = 0.0
        self.last_sdf_slice_axis = None
        self.last_sdf_slice_position = None
        self.last_sdf_slice_resolution = None
        self.last_sdf_grid_time = 0.0
        self.last_sdf_grid_resolution = None
        self.mapping_end = False

        self.thread_after_training_end = threading.Thread(target=self.after_training_end)

    def training_iteration_end_callback(self, trainer: Trainer):
        # send iteration results to GUI
        data_packet = GuiDataPacket()
        data_packet.time_stats = trainer.get_time_stats()
        # remove callback time from train_frame time
        train_frame_time = 0
        for key, t in data_packet.time_stats.items():
            if key == "train_frame":
                continue
            if key == "training_iteration":
                train_frame_time += t * trainer.num_iterations
            else:
                train_frame_time += t
        data_packet.time_stats["train_frame"] = train_frame_time
        data_packet.loss_stats = trainer.loss_dict
        data_packet.gpu_mem_usage = torch.cuda.memory_allocated() / (1024**3)  # in GB

        self.reply_gui(data_packet, must_reply=True)

    def training_frame_start_callback(self, trainer: Trainer, frame: Frame):
        # send new frame info to GUI
        data_packet = GuiDataPacket()
        data_packet.num_iterations = trainer.num_iterations
        data_packet.frame_idx = frame.get_frame_index()
        data_packet.frame_pose = frame.get_ref_pose().cpu().numpy()
        data_packet.scan_points = frame.get_points(to_world_frame=True, device="cpu").numpy()
        data_packet.key_frame_indices = [f.get_frame_index() for f in trainer.key_frame_set.frames]
        data_packet.selected_key_frame_indices = trainer.selected_key_frame_indices

        assert data_packet.key_frame_indices[-1] <= data_packet.frame_idx
        tqdm.write(f"[Training] Frame idx = {data_packet.frame_idx}")

        self.reply_gui(data_packet, must_reply=True)  # block until mapping is allowed to run

        return not self.control_packet.flag_gui_closed

    def training_end_callback(self, trainer: Trainer):
        data_packet = GuiDataPacket()
        data_packet.mapping_end = True
        self.queue_to_gui.put_nowait(data_packet)
        self.mapping_end = True
        self.thread_after_training_end.start()

    def after_training_end(self):
        tqdm.write("[After Training] Training finished. Waiting for GUI to close...")
        assert self.queue_from_gui is not None
        while True:
            if self.control_packet.flag_gui_closed:
                break

            if self.queue_from_gui.empty():
                time.sleep(0.001)
                continue

            data_packet = GuiDataPacket()
            data_packet.mapping_end = True
            self.reply_gui(data_packet, must_reply=True)

        tqdm.write("[After Training] GUI closed. Exiting...")

    def _get_control_packet_from_queue(self, get_latest: bool = True):
        if self.queue_from_gui is None:
            return None
        packet: Optional[GuiControlPacket] = None
        save_model_to_path = None
        while True and not self.queue_from_gui.empty():
            try:
                packet = self.queue_from_gui.get_nowait()
                if packet.save_model_to_path is not None and len(packet.save_model_to_path) > 0:
                    save_model_to_path = packet.save_model_to_path
                if not get_latest:
                    break
            except queue.Empty:
                break
        if packet is not None and save_model_to_path is not None:
            packet.save_model_to_path = save_model_to_path
        return packet

    def reply_gui(self, data_packet: Optional[GuiDataPacket] = None, must_reply: bool = False):
        if self.queue_from_gui is None:
            return
        if self.queue_from_gui.empty():
            if data_packet is not None and must_reply:
                self.queue_to_gui.put_nowait(data_packet)
            return

        while True:
            if self.queue_from_gui.empty():
                time.sleep(0.001)
                continue
            control_packet = self._get_control_packet_from_queue(get_latest=True)
            if control_packet is not None:
                self.control_packet = control_packet

            # reply to GUI if needed
            if data_packet is None:
                data_packet = GuiDataPacket()
            send_reply = must_reply

            # send octree if requested
            current_time = time.time()
            if (
                self.control_packet.octree_update_frequency > 0
                and current_time - self.last_octree_time > 1.0 / self.control_packet.octree_update_frequency
                and (self.control_packet.octree_min_size != self.last_octree_min_size or not self.mapping_end)
            ):
                self.last_octree_time = current_time
                self.last_octree_min_size = self.control_packet.octree_min_size

                mask1 = self.trainer.model.octree.voxels[:, -1] >= self.last_octree_min_size  # get valid voxels only
                mask2 = ~torch.all(self.trainer.model.octree.structure[:, :8] >= 0, dim=1)  # nodes without all children
                mask = mask1 & mask2
                data_packet.octree_voxel_centers = self.trainer.model.octree.voxel_centers[mask].cpu().numpy()  # (M, 3)
                data_packet.octree_voxel_sizes = (
                    (self.trainer.model.octree.voxels[mask, [-1]] * self.trainer.model.octree.cfg.resolution)
                    .cpu()
                    .numpy()
                )
                data_packet.octree_vertices = self.trainer.model.octree.vertex_indices[mask].cpu().numpy()
                data_packet.octree_little_endian_vertex_order = self.trainer.model.octree.little_endian_vertex_order
                data_packet.octree_resolution = self.trainer.model.octree.cfg.resolution

            # send SDF slice if requested
            current_time = time.time()
            if (
                self.control_packet.sdf_slice_frequency > 0
                and current_time - self.last_sdf_slice_time > 1.0 / self.control_packet.sdf_slice_frequency
                and (
                    self.control_packet.sdf_slice_axis != self.last_sdf_slice_axis
                    or self.control_packet.sdf_slice_position != self.last_sdf_slice_position
                    or self.control_packet.sdf_slice_resolution != self.last_sdf_slice_resolution
                    or not self.mapping_end
                )
            ):
                self.last_sdf_slice_time = current_time
                self.last_sdf_slice_axis = self.control_packet.sdf_slice_axis
                self.last_sdf_slice_position = self.control_packet.sdf_slice_position
                self.last_sdf_slice_resolution = self.control_packet.sdf_slice_resolution

                results = self.trainer.evaluater.extract_slice(
                    axis=self.control_packet.sdf_slice_axis,
                    pos=self.control_packet.sdf_slice_position,
                    resolution=self.control_packet.sdf_slice_resolution,
                    bound_min=self.gui_cfg.scene_bound_min,
                    bound_max=self.gui_cfg.scene_bound_max,
                )
                data_packet.sdf_slice_bounds = results["slice_bound"].cpu().tolist()
                data_packet.sdf_slice_axis = self.control_packet.sdf_slice_axis
                data_packet.sdf_slice_position = self.control_packet.sdf_slice_position
                data_packet.sdf_slice_resolution = self.control_packet.sdf_slice_resolution
                data_packet.sdf_slice = dict(
                    voxel_indices=results["voxel_indices"].cpu().numpy(),
                    sdf_prior=results["sdf_prior"].cpu().numpy(),
                    sdf=results["sdf"].cpu().numpy(),
                )
                send_reply = True

            # send SDF grid if requested
            current_time = time.time()
            if (
                self.control_packet.sdf_grid_frequency > 0
                and current_time - self.last_sdf_grid_time > 1.0 / self.control_packet.sdf_grid_frequency
                and (self.control_packet.sdf_grid_resolution != self.last_sdf_grid_resolution or not self.mapping_end)
            ):
                self.last_sdf_grid_time = current_time
                self.last_sdf_grid_resolution = self.control_packet.sdf_grid_resolution

                results = self.trainer.evaluater.extract_sdf_grid(
                    bound_min=self.gui_cfg.scene_bound_min,
                    bound_max=self.gui_cfg.scene_bound_max,
                    grid_resolution=self.control_packet.sdf_grid_resolution,
                )
                data_packet.sdf_grid_bounds = results["grid_bound"].cpu().tolist()
                data_packet.sdf_grid_resolution = self.control_packet.sdf_grid_resolution
                data_packet.sdf_grid = dict(
                    voxel_indices=results["voxel_indices"].cpu().numpy(),
                    sdf_prior=results["sdf_prior"].cpu().numpy(),
                    sdf=results["sdf"].cpu().numpy(),
                )
                send_reply = True

            # save model if requested
            model_path = self.control_packet.save_model_to_path
            if model_path is not None and len(model_path) > 0:
                self.trainer.save_model(model_path)
                self.control_packet.save_model_to_path = None
                data_packet.model_saved_path = model_path
                send_reply = True

            if send_reply:
                self.queue_to_gui.put_nowait(data_packet)
                data_packet = None  # only send once

            if self.control_packet.flag_mapping_run:
                break  # continue mapping

    def run(self):
        self.gui_process.start()
        self.control_packet = self.queue_from_gui.get(block=True)  # wait for GUI to be ready

        try:
            self.trainer.train()
        except KeyboardInterrupt:
            tqdm.write("KeyboardInterrupt detected. Stopping training...")
            data_packet = GuiDataPacket()
            data_packet.flag_exit = True
            self.reply_gui(data_packet, must_reply=True)
        finally:
            if self.thread_after_training_end.is_alive():
                tqdm.write("Waiting for after-training thread to finish...")
                self.thread_after_training_end.join()
            tqdm.write("Waiting for GUI to close...")
            self.gui_process.join()


def main():
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gui-config", type=str, help="path to GUI config file")
    parser.add_argument("--trainer-config", type=str, required=True, help="path to trainer config file")
    parser.add_argument("--gt-mesh-path", type=str, help="path to ground truth mesh file")
    parser.add_argument("--apply-offset-to-gt-mesh", action="store_true", help="apply scene offset to GT mesh")
    parser.add_argument(
        "--copy-scene-bound-to-gui",
        action="store_true",
        help="copy scene bounds from trainer config to GUI config",
    )
    args = parser.parse_args()

    if args.gui_config is not None:
        assert os.path.exists(args.gui_config), f"GUI config file {args.gui_config} does not exist"
        gui_cfg = GuiBaseConfig.from_yaml(args.gui_config)
    else:
        gui_cfg = GuiBaseConfig()
    if gui_cfg.view_file is not None and not os.path.isabs(gui_cfg.view_file):
        gui_cfg.view_file = os.path.join(os.path.dirname(args.gui_config), gui_cfg.view_file)
    if args.gt_mesh_path is not None and os.path.exists(args.gt_mesh_path):
        gui_cfg.gt_mesh_path = args.gt_mesh_path

    assert os.path.exists(args.trainer_config), f"Trainer config file {args.trainer_config} does not exist"

    trainer_cfg = TrainerConfig.from_yaml(args.trainer_config)
    trainer_cfg.profiling = True  # enable profiling for GUI
    if args.apply_offset_to_gt_mesh:
        offset = trainer_cfg.data.dataset_args.get("offset", None)
        if offset is not None:
            gui_cfg.gt_mesh_offset = offset
            tqdm.write(f"Applied offset {offset} to GT mesh in GUI.")
    if args.copy_scene_bound_to_gui:
        trainer_bound_min = trainer_cfg.model.residual_net_cfg.bound_min
        trainer_bound_max = trainer_cfg.model.residual_net_cfg.bound_max
        gui_cfg.scene_bound_min = trainer_bound_min
        gui_cfg.scene_bound_max = trainer_bound_max
        tqdm.write(f"Copied scene bounds from trainer config to GUI config: {trainer_bound_min}, {trainer_bound_max}")

    gui_trainer = GuiTrainer(gui_cfg, trainer_cfg)
    gui_trainer.run()


if __name__ == "__main__":
    main()
