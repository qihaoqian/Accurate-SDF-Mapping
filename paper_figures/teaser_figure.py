import numpy as np
import vedo
import cv2


def main():
    mesh: vedo.Mesh = vedo.load(
        "/home/daizhirui/D/GoogleDrive/Documents/UCSD/Research/ERL/SDF/Neural-SDF/reconstructed_mesh_result/room0-our.obj"
    )
    mesh.backface_culling(True).color("gray")
    mesh.scale(20)  # set scale according to sdf grid

    sdf_grid = np.load(
        "/home/daizhirui/D/GoogleDrive/Documents/UCSD/Research/ERL/SDF/Neural-SDF/sdf_grid_result/room0-voxblox.npy"
    )

    sdf_slice = sdf_grid[48]
    sdf_colored = (sdf_slice - sdf_slice.min()) / (sdf_slice.max() - sdf_slice.min()) * 255
    sdf_colored = sdf_colored.astype(np.uint8)
    sdf_colored = cv2.applyColorMap(sdf_colored, cv2.COLORMAP_JET)
    sdf_colored = cv2.cvtColor(sdf_colored, cv2.COLOR_BGR2RGB)

    img = vedo.Image(sdf_colored)
    pos = mesh.bounds()[::2]
    pos[0] = 20
    img.rotate_y(-90).pos(*pos)

    center = mesh.center_of_mass().tolist()
    center[0] -= sdf_colored.shape[0] // 2
    center[1] -= sdf_colored.shape[1] // 2
    print(center)

    traj = np.loadtxt("").reshape(-1, 4, 4)
    traj = traj[:, :3, 3]  # (N, 3)
    traj_lines: vedo.Lines = vedo.Lines(start_pts=traj[:-1], end_pts=traj[1:])
    traj_lines.color("red")
    traj_lines.linewidth(5)

    s = 2
    f = 2
    cam_frames = np.array(
        [
            [-s, -s, f],
            [s, -s, f],
            [s, s, f],
            [-s, s, f],
            [0, 0, 0],
            [-s, -s, f],
            [0, 0, 0],
            [s, -s, f],
            [0, 0, 0],
            [s, s, f],
            [0, 0, 0],
        ]
    )
    cam_frames_vedo: vedo.Lines = vedo.Lines(start_pts=cam_frames[:-1], end_pts=cam_frames[1:])
    cam_frames_vedo.color("blue").linewidth(2)

    vedo.show(
        mesh,
        img,
        size=(1440, 1440),
        axes=1,
        camera=dict(
            pos=(-0.856456, 1.59392, -3.71767),
            focal_point=(0.379982, 0.178761, -0.204810),
            viewup=(-0.950620, -0.116014, 0.287858),
            roll=97.1292,
            distance=3.98393,
            clipping_range=(0.0158820, 15.8820),
        ),
    )


if __name__ == "__main__":
    main()
