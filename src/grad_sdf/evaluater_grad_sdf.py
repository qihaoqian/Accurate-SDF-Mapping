import os
from typing import Dict, Callable

from grad_sdf.evaluater_base import EvaluaterBase, torch
from grad_sdf.model import SdfNetwork, SdfNetworkConfig, forward_with_batches


class GradSdfEvaluator(EvaluaterBase):
    """
    Evaluator for GradSDF models.
    """

    def __init__(
        self,
        batch_size: int,
        clean_mesh: bool = True,
        model_cfg: SdfNetworkConfig | None = None,
        model: torch.nn.Module | None = None,
        model_path: str | None = None,
        device: str = "cuda",
    ):
        """
        Args:
            batch_size: batch size for model inference
            clean_mesh: whether to clean mesh by removing voxels that are too big
            model_cfg: configuration for the SdfNetwork, required if model is not provided
            model: optional, if provided, use this model
            model_path: optional, if model is not provided, load the model from this path
            device: device to run the model on
        """
        self.batch_size = batch_size
        self.clean_mesh = clean_mesh
        self.model_cfg = model_cfg

        super().__init__(
            self.forward_model,
            model,
            model_path,
            self.create_model,
            device,
        )

        self.model: SdfNetwork

    def create_model(self, model_path: str) -> torch.nn.Module:
        """
        Create the model and load weights from the given path.
        Args:
            model_path: path to the model weights
        Returns:
            torch.nn.Module: the created model
        """
        assert self.model_cfg is not None
        model = SdfNetwork(self.model_cfg)
        model.to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def forward_model(self, model, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward the model with the given points.
        Args:
            model: the SdfNetwork model
            points: (..., 3) points in world coordinates
        Returns:
            dict of output tensors with keys:
                'voxel_indices': (...,) voxel indices for the points
                'sdf_prior': (...,) SDF prior from the octree
                'sdf_residual': (...,) SDF residual from the residual network
                'sdf': (...,) final SDF values (prior + residual)
        """
        voxel_indices, sdf_prior, sdf_residual, sdf = forward_with_batches(
            model,
            self.batch_size,
            points,
        )
        return dict(voxel_indices=voxel_indices, sdf_prior=sdf_prior, sdf_residual=sdf_residual, sdf=sdf)

    @torch.no_grad()
    def extract_mesh(
        self,
        bound_min: list[float],
        bound_max: list[float],
        grid_resolution: float,
        fields: list[str] = None,
        iso_value: float = 0.0,
        voxel_filter: Callable[[torch.Tensor, torch.Tensor, int, int], list] | None = None,
    ):
        if self.clean_mesh and voxel_filter is None:
            voxel_filter = self.model.voxel_filter_by_size

        if fields is None:
            fields = ["sdf_prior", "sdf_pred"]

        return super().extract_mesh(
            bound_min=bound_min,
            bound_max=bound_max,
            grid_resolution=grid_resolution,
            fields=fields,
            iso_value=iso_value,
            voxel_filter=voxel_filter,
        )


def main():
    import argparse

    from grad_sdf.trainer_config import TrainerConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--batch-size", type=int, default=20480)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    trainer_cfg = TrainerConfig.from_yaml(args.config)

    evaluator = GradSdfEvaluator(
        batch_size=args.batch_size,
        model_cfg=trainer_cfg.model,
        model_path=args.model_path,
        device=args.device,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.config), "eval")
    os.makedirs(output_dir, exist_ok=True)
    pass


if __name__ == "__main__":
    main()
