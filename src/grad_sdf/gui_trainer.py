import argparse
import asyncio
import multiprocessing as mp
import os
import threading
import time
from typing import Optional

from tqdm import tqdm

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
        self.replied_sdf_slice_for_steps = set()
        self.replied_sdf_grid_for_steps = set()

        self.thread_after_training_end = threading.Thread(target=self.after_training_end)

    def training_iteration_end_callback(self, trainer: Trainer):
        # send iteration results to GUI
        data_packet = GuiDataPacket()
        data_packet.time_stats = trainer.get_time_stats()
        data_packet.loss_stats = trainer.loss_dict

        self.reply_gui()

    def training_frame_start_callback(self, trainer: Trainer, frame: Frame):
        # send new frame info to GUI
        data_packet = GuiDataPacket()
        data_packet.frame_idx = frame.get_frame_index()
        data_packet.frame_pose = frame.get_ref_pose().cpu().numpy()
        data_packet.scan_points = frame.get_points(to_world_frame=True, device="cpu").numpy()
        data_packet.key_frame_indices = [f.get_frame_index() for f in trainer.key_frame_set.frames]
        data_packet.selected_key_frame_indices = trainer.selected_key_frame_indices

        data_packet.octree_voxel_centers = trainer.model.octree.voxel_centers.cpu().numpy()
        data_packet.octree_voxel_sizes = (
            (trainer.model.octree.voxels[:, [-1]] * trainer.model.octree.cfg.resolution).cpu().numpy()
        )
        data_packet.octree_vertices = trainer.model.octree.vertex_indices.cpu().numpy()
        data_packet.octree_little_endian_vertex_order = trainer.model.octree.little_endian_vertex_order

        self.reply_gui()  # block until mapping is allowed to run

    def training_end_callback(self, trainer: Trainer):
        data_packet = GuiDataPacket()
        data_packet.mapping_end = True
        self.queue_to_gui.put_nowait(data_packet)
        self.thread_after_training_end.start()

    def after_training_end(self):
        tqdm.write("Training finished. Waiting for GUI to close...")
        assert self.queue_from_gui is not None
        while True:
            if self.queue_from_gui.empty():
                time.sleep(0.001)
                continue

            data_packet = GuiDataPacket()
            data_packet.mapping_end = True
            self.reply_gui(data_packet)

            if self.control_packet.flag_gui_closed:
                break

        tqdm.write("GUI closed. Exiting...")

    def _get_control_packet_from_queue(self, get_latest: bool = True):
        if self.queue_from_gui is None:
            return None
        packet: Optional[GuiControlPacket] = None
        save_model_to_path = None
        while True and not self.queue_from_gui.empty():
            try:
                packet_new = self.queue_from_gui.get_nowait()
                if packet is not None:
                    del packet
                packet = packet_new
                if packet.save_model_to_path is not None and len(packet.save_model_to_path) > 0:
                    save_model_to_path = packet.save_model_to_path
                if not get_latest:
                    break
            except asyncio.QueueEmpty:
                break
        if packet is not None and save_model_to_path is not None:
            packet.save_model_to_path = save_model_to_path
        return packet

    def reply_gui(self, data_packet: Optional[GuiDataPacket] = None):
        if self.queue_from_gui is None:
            return
        if self.queue_from_gui.empty():
            return

        control_packet = self._get_control_packet_from_queue(get_latest=True)
        if control_packet is not None:
            self.control_packet = control_packet

        # block if GUI says the mapping should pause
        while not self.control_packet.flag_mapping_run:
            time.sleep(0.001)
            if self.queue_from_gui.empty():
                continue
            control_packet = self._get_control_packet_from_queue(get_latest=True)
            if control_packet is not None:
                self.control_packet = control_packet

            # reply to GUI if needed
            if data_packet is None:
                data_packet = GuiDataPacket()
            send_reply = False
            #
            if (
                self.control_packet.sdf_slice_frequency > 0
                and self.trainer.global_step % self.control_packet.sdf_slice_frequency == 0
                and self.trainer.global_step not in self.replied_sdf_slice_for_steps
            ):
                self.replied_sdf_slice_for_steps.add(self.trainer.global_step)
                resolution = 1.0 / self.control_packet.sdf_slice_resolution
                results = self.trainer.evaluater.extract_slice(
                    axis=self.control_packet.sdf_slice_axis,
                    pos=self.control_packet.sdf_slice_position,
                    resolution=resolution,
                    bound_min=self.gui_cfg.scene_bound_min,
                    bound_max=self.gui_cfg.scene_bound_max,
                )
                data_packet.sdf_slice_bounds = results["slice_bound"].cpu().tolist()
                data_packet.sdf_slice_axis = self.control_packet.sdf_slice_axis
                data_packet.sdf_slice_position = self.control_packet.sdf_slice_position
                data_packet.sdf_slice_resolution = resolution
                data_packet.sdf_slice = dict(
                    voxel_indices=results["voxel_indices"].cpu().numpy(),
                    sdf_prior=results["sdf_prior"].cpu().numpy(),
                    sdf=results["sdf"].cpu().numpy(),
                )
                send_reply = True

            if (
                self.control_packet.sdf_grid_frequency > 0
                and self.trainer.global_step % self.control_packet.sdf_grid_frequency == 0
                and self.trainer.global_step not in self.replied_sdf_grid_for_steps
            ):
                self.replied_sdf_grid_for_steps.add(self.trainer.global_step)
                resolution = 1.0 / self.control_packet.sdf_grid_resolution
                results = self.trainer.evaluater.extract_sdf_grid(
                    bound_min=self.gui_cfg.scene_bound_min,
                    bound_max=self.gui_cfg.scene_bound_max,
                    grid_resolution=resolution,
                )
                data_packet.sdf_grid_bounds = results["grid_bound"].cpu().tolist()
                data_packet.sdf_grid_resolution = resolution
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

    def run(self):
        self.gui_process.start()
        self.control_packet = self.queue_from_gui.get(block=True)  # wait for GUI to be ready

        try:
            self.trainer.train()
        except KeyboardInterrupt:
            tqdm.write("KeyboardInterrupt detected. Stopping training...")
            data_packet = GuiDataPacket()
            data_packet.flag_exit = True
            self.reply_gui(data_packet)
        finally:
            tqdm.write("Waiting for GUI to close...")
            self.thread_after_training_end.join()
            self.gui_process.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui-config", type=str, help="path to GUI config file")
    parser.add_argument("--trainer-config", type=str, required=True, help="path to trainer config file")
    parser.add_argument("--gt-mesh-path", type=str, help="path to ground truth mesh file")
    args = parser.parse_args()

    if args.gui_config is not None:
        assert os.path.exists(args.gui_config), f"GUI config file {args.gui_config} does not exist"
        gui_cfg = GuiBaseConfig.from_yaml(args.gui_config)
    else:
        gui_cfg = GuiBaseConfig()
    if args.gt_mesh_path is not None and os.path.exists(args.gt_mesh_path):
        gui_cfg.gt_mesh_path = args.gt_mesh_path

    assert os.path.exists(args.trainer_config), f"Trainer config file {args.trainer_config} does not exist"

    trainer_cfg = TrainerConfig.from_yaml(args.trainer_config)

    gui_trainer = GuiTrainer(gui_cfg, trainer_cfg)
    gui_trainer.run()


if __name__ == "__main__":
    main()
