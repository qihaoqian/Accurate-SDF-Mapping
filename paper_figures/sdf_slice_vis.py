import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

from demo.parser import get_parser


def get_points(bound, res, offset):
    # Convert bound to numpy array and apply offset
    bound = np.array(bound) - offset
    x_min, x_max = bound[0]
    y_min, y_max = bound[1]
    z_min, z_max = bound[2]

    # Calculate the range for each dimension
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_res = int(x_range * res) + 1
    y_res = int(y_range * res) + 1
    z_res = int(z_range * res) + 1

    x = torch.linspace(x_min, x_max, x_res)
    y = torch.linspace(y_min, y_max, y_res)
    z = torch.linspace(z_min, z_max, z_res)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)

    return points, (x_res, y_res, z_res)


def visualize_sdf_slice(
    sdf_grid,
    slice_idx,
    slice_axis,
    *,
    x_vals=None,
    y_vals=None,
    z_vals=None,
    save_dir=".",
    filename=None,
    title=None,
    cmap="jet",
    vlim=None,  # 手动设定[-vlim, vlim]，不设则自动对称
    show=False,  # 需要屏幕显示就设 True
    smin=None,
    smax=None,
):
    """
    可视化 SDF 体数据在某一轴向上的二维切片。

    参数
    ----
    sdf_grid : (Nx, Ny, Nz) numpy array
    slice_idx : int，切片索引
    slice_axis : {"x","y","z"}，切片轴
    x_vals, y_vals, z_vals : 1D array，可选；对应各轴坐标（米）。不传则用索引 0..N-1
    save_dir : 保存目录
    filename : 保存文件名；不传则自动生成
    title : 图标题；不传则自动生成
    cmap : 颜色图
    vlim : 颜色轴范围的正半幅（对称到 [-vlim, vlim]）
    show : 是否调用 plt.show()
    """
    # 1) 基础检查与坐标向量准备
    assert slice_axis in {"x", "y", "z"}, "slice_axis 必须是 'x'/'y'/'z'"
    Nx, Ny, Nz = sdf_grid.shape

    if x_vals is None:
        x_vals = np.arange(Nx)
    if y_vals is None:
        y_vals = np.arange(Ny)
    if z_vals is None:
        z_vals = np.arange(Nz)

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color="white")

    # 2) 取切片 & 确定绘图的两个坐标轴（横轴=第二维，纵轴=第一维）
    if slice_axis == "x":
        assert 0 <= slice_idx < Nx, "slice_idx 越界"
        sdf_slice = sdf_grid[slice_idx, :, :]  # (Ny, Nz)
        coord1_vals, coord2_vals = y_vals, z_vals  # 行对应 y，列对应 z
        coord_names = ("y", "z")
    elif slice_axis == "y":
        assert 0 <= slice_idx < Ny, "slice_idx 越界"
        sdf_slice = sdf_grid[:, slice_idx, :]  # (Nx, Nz)
        coord1_vals, coord2_vals = x_vals, z_vals  # 行对应 x，列对应 z
        coord_names = ("x", "z")
    else:  # "z"
        assert 0 <= slice_idx < Nz, "slice_idx 越界"
        sdf_slice = sdf_grid[:, :, slice_idx]  # (Nx, Ny)
        coord1_vals, coord2_vals = x_vals, y_vals  # 行对应 x，列对应 y
        coord_names = ("x", "y")

    # 3) 生成网格（注意：contour 默认 'xy' 语义 => X=横轴(列)，Y=纵轴(行)）
    X, Y = np.meshgrid(coord2_vals, coord1_vals, indexing="xy")  # X 对应列，Y 对应行

    # 4) 颜色范围：对称到 0
    if smin is None:
        smin = float(np.min(sdf_slice))
    if smax is None:
        smax = float(np.max(sdf_slice))
    print(f"SDF range: [{smin:.4f}, {smax:.4f}]")
    if vlim is None:
        vlim = max(abs(smin), abs(smax)) if (smin != 0 or smax != 0) else 1.0

    # 5) 绘图
    plt.figure(figsize=(12, 10))

    # imshow: 数据 shape=(n行, n列) => (len(coord1), len(coord2))
    # extent: [xmin, xmax, ymin, ymax] = [横轴min,max, 纵轴min,max]
    im = plt.imshow(
        sdf_slice,
        extent=[coord2_vals[0], coord2_vals[-1], coord1_vals[0], coord1_vals[-1]],
        origin="lower",
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        aspect="equal",
    )

    # 等高线：用与 imshow 一致的 X(横) / Y(纵)
    # 自动给出一组等高线（含 0 等高），也可按需改 levels
    # 这里采用对称 levels，保证包含零等高线
    n_levels = 10
    levels = np.linspace(-vlim, vlim, n_levels)
    cs = plt.contour(X, Y, sdf_slice, levels=levels, colors="k", alpha=0.25, linewidths=0.6)

    # 0 等高线高亮（表面）
    zero_cs = plt.contour(X, Y, sdf_slice, levels=[0.0], colors="red", linewidths=1.8)

    cbar = plt.colorbar(im, shrink=0.6)
    cbar.ax.tick_params(labelsize=21)
    plt.gca().tick_params(axis="both", which="major", labelsize=21)
    plt.tight_layout()
    # cbar.set_label("SDF Value")

    # 轴标签与标题
    # plt.xlabel(f"{coord_names[1]} (m)", fontsize=12)  # 横轴是第二个坐标
    # plt.ylabel(f"{coord_names[0]} (m)", fontsize=12)  # 纵轴是第一个坐标
    # if title is None:
    #     title = f"SDF slice @ {slice_axis}={slice_idx}"
    # plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    # # 辅助信息
    # plt.suptitle(
    #     f"Range: [{smin:.4f}, {smax:.4f}] | Grid: {sdf_slice.shape[0]}×{sdf_slice.shape[1]} | vlim=±{vlim:.4f}",
    #     fontsize=10, y=0.95
    # )

    # 6) 保存/显示
    if filename is None:
        filename = f"sdf_slice_{slice_axis}{slice_idx}.png"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close()

    # 7) 日志
    print(f"SDF slice saved to: {save_path}")
    print(f"  Slice index: {slice_idx} (axis {slice_axis})")
    print(f"  SDF range: [{smin:.4f}, {smax:.4f}]")
    print(f"  Grid size: {sdf_slice.shape[0]}×{sdf_slice.shape[1]}")


def main():
    # parser = get_parser()
    # args = parser.parse_args()
    # ckpt_path = os.path.join(args.log_dir, args.exp_name, "ckpt", "final_ckpt.pth")
    # save_dir = os.path.join(args.log_dir, args.exp_name, "misc")
    # mapper, decoder = load_checkpoint(ckpt_path, args)
    # points, (x_res, y_res, z_res) = get_points([[2.47, 5.57], [1.52, 6.52], [0.0, 8.04]], 80, -10+4.02)
    # print(f"Points shape: {points.shape}",points.min(axis=0),points.max(axis=0))
    # sdf_pred, sdf_prior, _, valid_mask, _, _ = predict_sdf(mapper, decoder, points.cuda())
    # sdf_pred_grid = torch.full((points.shape[0],), float('nan')).cuda()
    # sdf_pred_grid[valid_mask] = sdf_pred
    # sdf_prior_grid = torch.full((points.shape[0],), float('nan')).cuda()
    # sdf_prior_grid[valid_mask] = sdf_prior
    # sdf_pred_grid = sdf_pred_grid.cpu().numpy().reshape(x_res, y_res, z_res)
    # sdf_prior_grid = sdf_prior_grid.cpu().numpy().reshape(x_res, y_res, z_res)
    # np.save(os.path.join(save_dir, "sdf_pred_grid.npy"), sdf_pred_grid)
    # np.save(os.path.join(save_dir, "sdf_prior_grid.npy"), sdf_prior_grid)
    sdf_pred_grid = np.load(
        "/home/daizhirui/D/GoogleDrive/Documents/UCSD/Research/ERL/SDF/Neural-SDF/sdf_grid_result/room0-voxblox.npy"
    )
    print(sdf_pred_grid.shape)
    visualize_sdf_slice(
        sdf_pred_grid,
        48,
        "x",
        x_vals=np.linspace(2.47, 5.57, sdf_pred_grid.shape[0], endpoint=False),
        y_vals=np.linspace(1.52, 6.52, sdf_pred_grid.shape[1], endpoint=False),
        z_vals=np.linspace(0.0, 8.04, sdf_pred_grid.shape[2], endpoint=False),
        save_dir=script_dir,
        filename=f"voxblox-sdf-slice-room0.png",
        title="SDF Pred",
        smin=-0.9948,
        smax=0.5117,
    )


if __name__ == "__main__":
    main()
