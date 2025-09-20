import argparse
import os.path

import vedo
import importlib.util


def visualize_mesh(
    mesh_fp: str,
    cam_config_fp: str,
    lighting_config_fp: str,
    show_lights: bool,
    output_img_fp: str,
    interactive: bool,
    image_size: tuple = (2560, 1440),
):
    mesh: vedo.Mesh = vedo.load(mesh_fp).color("gray").backface_culling(True)
    # mesh.lighting(ambient=0.8, diffuse=1.0)
    if os.path.exists(cam_config_fp):
        # evaluate the python file
        spec = importlib.util.spec_from_file_location("cam_config", cam_config_fp)
        cam_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cam_config_module)  # type: ignore
        if hasattr(cam_config_module, "cam_config"):
            cam_config = cam_config_module.cam_config
        else:
            cam_config = None
    else:
        cam_config = None

    lights = []
    light_spheres = []
    if os.path.exists(lighting_config_fp):
        # evaluate the python file
        spec = importlib.util.spec_from_file_location("lighting_config", lighting_config_fp)
        lighting_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lighting_config_module)  # type: ignore
        if hasattr(lighting_config_module, "lighting_config"):
            lighting_config = lighting_config_module.lighting_config
            for light_cfg in lighting_config:
                light = vedo.Light(**light_cfg)
                lights.append(light)
                if show_lights:
                    sphere = vedo.Sphere(light_cfg["pos"], r=0.05).color("yellow").lighting("off")
                    light_spheres.append(sphere)
    os.makedirs(os.path.dirname(output_img_fp), exist_ok=True)
    vedo.show(
        mesh,
        *lights,
        *light_spheres,
        size=image_size,
        camera=cam_config,
        screenshot=output_img_fp,
        interactive=interactive,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-fp", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--cam-config-fp", type=str, help="Path to the camera configuration file.")
    parser.add_argument("--lighting-config-fp", type=str, help="Path to the lighting configuration file.")
    parser.add_argument("--show-lights", action="store_true", help="Whether to show the light sources.")
    parser.add_argument("--output-img-fp", type=str, required=True, help="Path to save the output image.")
    parser.add_argument("--interactive", action="store_true", help="Whether to show the interactive window.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(2560, 1440), help="Size of the output image.")
    args = parser.parse_args()
    visualize_mesh(
        args.mesh_fp,
        args.cam_config_fp,
        args.lighting_config_fp,
        args.show_lights,
        args.output_img_fp,
        args.interactive,
        tuple(args.image_size),
    )


if __name__ == "__main__":
    main()
