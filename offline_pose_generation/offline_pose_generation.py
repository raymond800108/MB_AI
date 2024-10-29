# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Generate a [YCBVideo, DOPE] synthetic datasets
"""

import argparse
import os
import signal

import carb
import numpy as np
import torch
import yaml
from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser("Pose Generation data generator")
parser.add_argument("--num_mesh", type=int, default=30, help="Number of frames to record similar to MESH dataset")
parser.add_argument("--num_dome", type=int, default=30, help="Number of frames to record similar to DOME dataset")
parser.add_argument(
    "--dome_interval",
    type=int,
    default=1,
    help="Number of frames to capture before switching DOME background. When generating large datasets, increasing this interval will reduce time taken. A good value to set is 10.",
)
parser.add_argument("--output_folder", "-o", type=str, default="output", help="Output directory.")
parser.add_argument("--use_s3", action="store_true", help="Saves output to s3 bucket. Only supported by DOPE writer.")
parser.add_argument(
    "--bucket",
    type=str,
    default=None,
    help="Bucket name to store output in. See naming rules: https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html",
)
parser.add_argument("--s3_region", type=str, default="us-east-1", help="s3 region.")
parser.add_argument("--endpoint", type=str, default=None, help="s3 endpoint to write to.")
parser.add_argument(
    "--writer",
    type=str,
    default="DOPE",
    help="Which writer to use to output data. Choose between: [YCBVideo, DOPE]",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Generates data for testing. Hardcodes the pose of the object to compare output data with expected data to ensure that generation is correct.",
)

args, unknown_args = parser.parse_known_args()

# Do not write to s3 if in test mode
if args.test:
    args.use_s3 = False

if args.use_s3 and (args.endpoint is None or args.bucket is None):
    raise Exception("To use s3, --endpoint and --bucket must be specified.")

CONFIG_FILES = {"dope": "config/dope_config.yaml", "ycbvideo": "config/ycb_config.yaml"}
TEST_CONFIG_FILES = {"dope": "tests/dope/test_dope_config.yaml", "ycbvideo": "tests/ycbvideo/test_ycb_config.yaml"}

# Path to config file:
cf_map = TEST_CONFIG_FILES if args.test else CONFIG_FILES
CONFIG_FILE = cf_map[args.writer.lower()]

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)

with open(CONFIG_FILE_PATH) as f:
    config_data = yaml.full_load(f)

OBJECTS_TO_GENERATE = config_data["OBJECTS_TO_GENERATE"]

kit = SimulationApp(launch_config=config_data["CONFIG"])

import math

import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.replicator.isaac.scripts.writers import DOPEWriter, YCBVideoWriter

# Since the simulation is mostly collision checking, a larger physics dt can be used to speed up the object movements
world = World(physics_dt=1.0 / 30.0)
world.reset()

from flying_distractors.collision_box import CollisionBox
from flying_distractors.dynamic_object import DynamicObject
from flying_distractors.dynamic_object_set import DynamicObjectSet
from flying_distractors.dynamic_shape_set import DynamicShapeSet
from flying_distractors.flying_distractors import FlyingDistractors
from omni.isaac.core.utils.random import get_random_world_pose_in_view
from omni.isaac.core.utils.transformations import get_world_pose_from_relative
from tests.test_utils import clean_output_dir, run_pose_generation_test


class RandomScenario(torch.utils.data.IterableDataset):
    def __init__(
        self,
        num_mesh,
        num_dome,
        dome_interval,
        output_folder,
        use_s3=False,
        endpoint="",
        s3_region="us-east-1",
        writer="dope",
        bucket="",
        test=False,
    ):
        self.test = test

        if writer == "ycbvideo":
            self.writer_helper = YCBVideoWriter
        elif writer == "dope":
            self.writer_helper = DOPEWriter
        else:
            raise Exception("Invalid writer specified. Choose between [DOPE, YCBVideo].")

        self.result = True
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.result = False
            return
        self.dome_texture_path = assets_root_path + "/NVIDIA/Assets/Skies/"
        self.ycb_asset_path = assets_root_path + "/Isaac/Props/YCB/Axis_Aligned/"
        self.asset_path = assets_root_path + "/Isaac/Props/YCB/Axis_Aligned/"

        self.train_parts = []
        self.train_part_mesh_path_to_prim_path_map = {}
        self.mesh_distractors = FlyingDistractors()
        self.dome_distractors = FlyingDistractors()
        self.current_distractors = None

        self.data_writer = None
        self.num_mesh = max(0, num_mesh) if not self.test else 5
        self.num_dome = max(0, num_dome) if not self.test else 0
        self.train_size = self.num_mesh + self.num_dome
        self.dome_interval = dome_interval

        self._output_folder = output_folder if use_s3 else os.path.join(os.getcwd(), output_folder)
        self.use_s3 = use_s3
        self.endpoint = endpoint
        self.s3_region = s3_region
        self.bucket = bucket

        self.writer_config = {
            "output_folder": self._output_folder,
            "use_s3": self.use_s3,
            "bucket_name": self.bucket,
            "endpoint_url": self.endpoint,
            "s3_region": self.s3_region,
            "train_size": self.train_size,
        }

        self._setup_world()

        self.cur_idx = 0
        self.exiting = False
        self.last_frame_reached = False

        # Clean up output folder ahead of test
        if not self.use_s3 and self.test:
            clean_output_dir(self._output_folder)

        # Disable capture on play and async rendering
        self._carb_settings = carb.settings.get_settings()
        self._carb_settings.set("/omni/replicator/captureOnPlay", False)
        self._carb_settings.set("/app/asyncRendering", False)
        self._carb_settings.set("/omni/replicator/asyncRendering", False)

        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, *args, **kwargs):
        print("Exiting dataset generation..")
        self.exiting = True

    def _setup_world(self):
        """Populate scene with assets and prepare for synthetic data generation."""
        self._setup_camera()

        rep.settings.set_render_rtx_realtime()

        # Allow flying distractors to float
        world.get_physics_context().set_gravity(0.0)

        collision_box = self._setup_collision_box()

        world.scene.add(collision_box)

        self._setup_distractors(collision_box)

        self._setup_train_objects()

        if not self.test:
            self._setup_randomizers()

        # Update the app a few times to make sure the materials are fully loaded and world scene objects are registered
        for _ in range(5):
            kit.app.update()

        # Setup writer
        self.writer_helper.register_pose_annotator(config_data=config_data)
        self.writer = self.writer_helper.setup_writer(config_data=config_data, writer_config=self.writer_config)
        self.writer.attach([self.render_product])

        self.dome_distractors.set_visible(False)

        # Generate the replicator graphs without triggering any writing
        rep.orchestrator.preview()

    def _setup_camera(self):
        focal_length = config_data["HORIZONTAL_APERTURE"] * config_data["F_X"] / config_data["WIDTH"]

        # Setup camera and render product
        self.camera = rep.create.camera(
            position=(0, 0, 0),
            rotation=np.array(config_data["CAMERA_RIG_ROTATION"]),
            focal_length=focal_length,
            clipping_range=(0.01, 10000),
        )

        self.render_product = rep.create.render_product(self.camera, (config_data["WIDTH"], config_data["HEIGHT"]))

        camera_rig_path = str(rep.utils.get_node_targets(self.camera.node, "inputs:primsIn")[0])
        self.camera_path = camera_rig_path + "/Camera"

        with rep.get.prims(prim_types=["Camera"]):
            rep.modify.pose(
                rotation=rep.distribution.uniform(
                    np.array(config_data["CAMERA_ROTATION"]), np.array(config_data["CAMERA_ROTATION"])
                )
            )

        self.rig = XFormPrim(prim_path=camera_rig_path)

    def _setup_collision_box(self):
        # Create a collision box in view of the camera, allowing distractors placed in the box to be within
        # [MIN_DISTANCE, MAX_DISTANCE] of the camera. The collision box will be placed in front of the camera,
        # regardless of CAMERA_ROTATION or CAMERA_RIG_ROTATION.

        self.fov_x = 2 * math.atan(config_data["WIDTH"] / (2 * config_data["F_X"]))
        self.fov_y = 2 * math.atan(config_data["HEIGHT"] / (2 * config_data["F_Y"]))
        theta_x = self.fov_x / 2.0
        theta_y = self.fov_y / 2.0

        # Collision box dimensions lower than 1.3 do not work properly
        collision_box_width = max(2 * config_data["MAX_DISTANCE"] * math.tan(theta_x), 1.3)
        collision_box_height = max(2 * config_data["MAX_DISTANCE"] * math.tan(theta_y), 1.3)
        collision_box_depth = config_data["MAX_DISTANCE"] - config_data["MIN_DISTANCE"]

        collision_box_path = "/World/collision_box"
        collision_box_name = "collision_box"

        # Collision box is centered between MIN_DISTANCE and MAX_DISTANCE, with translation relative to camera in the z
        # direction being negative due to cameras in Isaac Sim having coordinates of -z out, +y up, and +x right.
        collision_box_translation_from_camera = np.array(
            [0, 0, (config_data["MIN_DISTANCE"] + config_data["MAX_DISTANCE"]) / 2.0]
        )

        # Collision box has no rotation with respect to the camera.
        collision_box_rotation_from_camera = np.array([0, 0, 0])
        collision_box_orientation_from_camera = euler_angles_to_quat(collision_box_rotation_from_camera, degrees=True)

        # Get the desired pose of the collision box from a pose defined locally with respect to the camera.
        camera_prim = world.stage.GetPrimAtPath(self.camera_path)
        collision_box_center, collision_box_orientation = get_world_pose_from_relative(
            camera_prim, collision_box_translation_from_camera, collision_box_orientation_from_camera
        )

        return CollisionBox(
            collision_box_path,
            collision_box_name,
            position=collision_box_center,
            orientation=collision_box_orientation,
            width=collision_box_width,
            height=collision_box_height,
            depth=collision_box_depth,
        )

    def _setup_distractors(self, collision_box):
        # List of distractor objects should not contain objects that are being used for training
        train_objects = [object["part_name"] for object in OBJECTS_TO_GENERATE]
        distractor_mesh_filenames = [
            file_name for file_name in config_data["MESH_FILENAMES"] if file_name not in train_objects
        ]

        usd_path_list = [
            f"{self.ycb_asset_path}{usd_filename_prefix}.usd" for usd_filename_prefix in distractor_mesh_filenames
        ]
        mesh_list = [f"_{usd_filename_prefix[1:]}" for usd_filename_prefix in distractor_mesh_filenames]

        if self.num_mesh > 0:
            # Distractors for the MESH dataset
            mesh_shape_set = DynamicShapeSet(
                "/World/mesh_shape_set",
                "mesh_shape_set",
                "mesh_shape",
                "mesh_shape",
                config_data["NUM_MESH_SHAPES"],
                collision_box,
                scale=np.array(config_data["SHAPE_SCALE"]),
                mass=config_data["SHAPE_MASS"],
                fraction_glass=config_data["MESH_FRACTION_GLASS"],
            )
            self.mesh_distractors.add(mesh_shape_set)

            mesh_object_set = DynamicObjectSet(
                "/World/mesh_object_set",
                "mesh_object_set",
                usd_path_list,
                mesh_list,
                "mesh_object",
                "mesh_object",
                config_data["NUM_MESH_OBJECTS"],
                collision_box,
                scale=np.array(config_data["OBJECT_SCALE"]),
                mass=config_data["OBJECT_MASS"],
                fraction_glass=config_data["MESH_FRACTION_GLASS"],
            )
            self.mesh_distractors.add(mesh_object_set)
            # Set the current distractors to the mesh dataset type
            self.current_distractors = self.mesh_distractors

        if self.num_dome > 0:
            # Distractors for the DOME dataset
            dome_shape_set = DynamicShapeSet(
                "/World/dome_shape_set",
                "dome_shape_set",
                "dome_shape",
                "dome_shape",
                config_data["NUM_DOME_SHAPES"],
                collision_box,
                scale=np.array(config_data["SHAPE_SCALE"]),
                mass=config_data["SHAPE_MASS"],
                fraction_glass=config_data["DOME_FRACTION_GLASS"],
            )
            self.dome_distractors.add(dome_shape_set)

            dome_object_set = DynamicObjectSet(
                "/World/dome_object_set",
                "dome_object_set",
                usd_path_list,
                mesh_list,
                "dome_object",
                "dome_object",
                config_data["NUM_DOME_OBJECTS"],
                collision_box,
                scale=np.array(config_data["OBJECT_SCALE"]),
                mass=config_data["OBJECT_MASS"],
                fraction_glass=config_data["DOME_FRACTION_GLASS"],
            )
            self.dome_distractors.add(dome_object_set)

    def _setup_train_objects(self):
        # Add the part to train the network on
        train_part_idx = 0
        for object in OBJECTS_TO_GENERATE:
            for prim_idx in range(object["num"]):
                part_name = object["part_name"]
                ref_path = self.asset_path + part_name + ".usd"
                prim_type = object["prim_type"]


                if self.writer_helper == YCBVideoWriter and prim_type not in config_data["CLASS_NAME_TO_INDEX"]:
                    raise Exception(f"Train object {prim_type} is not in CLASS_NAME_TO_INDEX in config.yaml.")

                path = "/World/" + prim_type + f"_{prim_idx}"
                path = "/World/" + prim_type

                mesh_path = path + "/" + prim_type
                name = f"train_part_{train_part_idx}"

                self.train_part_mesh_path_to_prim_path_map[mesh_path] = path

                train_part = DynamicObject(
                    usd_path=ref_path,
                    prim_path=path,
                    mesh_path=mesh_path,
                    name=name,
                    position=np.array([0.0, 0.0, 0.0]),
                    scale=config_data["OBJECT_SCALE"],
                    mass=1.0,
                )

                train_part.prim.GetAttribute("physics:rigidBodyEnabled").Set(True)

                self.train_parts.append(train_part)

                # Add semantic information
                mesh_prim = world.stage.GetPrimAtPath(mesh_path)
                add_update_semantics(mesh_prim, prim_type)

                train_part_idx += 1

                if prim_idx == 0 and self.writer_helper == YCBVideoWriter:
                    # Save the vertices of the part in '.xyz' format. This will be used in one of PoseCNN's loss functions
                    coord_prim = world.stage.GetPrimAtPath(path)
                    self.writer_helper.save_mesh_vertices(mesh_prim, coord_prim, prim_type, self._output_folder)

    def _setup_randomizers(self):
        """Add domain randomization with Replicator Randomizers"""
        # Create and randomize sphere lights
        def randomize_sphere_lights():
            lights = rep.create.light(
                light_type="Sphere",
                color=rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                intensity=rep.distribution.uniform(100000, 3000000),
                position=rep.distribution.uniform((-250, -250, -250), (250, 250, 100)),
                scale=rep.distribution.uniform(1, 20),
                count=config_data["NUM_LIGHTS"],
            )
            return lights.node

        # Randomize prim colors
        def randomize_colors(prim_path_regex):
            prims = rep.get.prims(path_pattern=prim_path_regex)
            mats = rep.create.material_omnipbr(
                metallic=rep.distribution.uniform(0.0, 1.0),
                roughness=rep.distribution.uniform(0.0, 1.0),
                diffuse=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
                count=100,
            )
            with prims:
                rep.randomizer.materials(mats)
            return prims.node

        rep.randomizer.register(randomize_sphere_lights, override=True)
        rep.randomizer.register(randomize_colors, override=True)

        with rep.trigger.on_frame():
            rep.randomizer.randomize_sphere_lights()
            rep.randomizer.randomize_colors("(?=.*shape)(?=.*nonglass).*")

    def _setup_dome_randomizers(self):
        """Add domain randomization with Replicator Randomizers"""

        # Create and randomize a dome light for the DOME dataset
        def randomize_domelight(texture_paths):
            lights = rep.create.light(
                light_type="Dome",
                rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                texture=rep.distribution.choice(texture_paths),
            )
            return lights.node

        rep.randomizer.register(randomize_domelight, override=True)

        dome_texture_paths = [
            self.dome_texture_path + dome_texture + ".hdr" for dome_texture in config_data["DOME_TEXTURES"]
        ]

        with rep.trigger.on_frame(interval=self.dome_interval):
            rep.randomizer.randomize_domelight(dome_texture_paths)

    def randomize_movement_in_view(self, prim):
        """Randomly move and rotate prim such that it stays in view of camera.

        Args:
            prim (DynamicObject): prim to randomly move and rotate.
        """
        if not self.test:
            camera_prim = world.stage.GetPrimAtPath(self.camera_path)
            rig_prim = world.stage.GetPrimAtPath(self.rig.prim_path)
            translation, orientation = get_random_world_pose_in_view(
                camera_prim,
                config_data["MIN_DISTANCE"],
                config_data["MAX_DISTANCE"],
                self.fov_x,
                self.fov_y,
                config_data["FRACTION_TO_SCREEN_EDGE"],
                rig_prim,
                np.array(config_data["MIN_ROTATION_RANGE"]),
                np.array(config_data["MAX_ROTATION_RANGE"]),
            )
        else:
            translation, orientation = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])

        prim.set_world_pose(translation, orientation)

    def __iter__(self):
        return self

    def __next__(self):
        # First frame of DOME dataset
        if self.cur_idx == self.num_mesh:  # MESH datset generation complete, switch to DOME dataset
            print(f"Starting DOME dataset generation of {self.num_dome} frames..")

            # Hide the FlyingDistractors used for the MESH dataset
            self.mesh_distractors.set_visible(False)

            # Show the FlyingDistractors used for the DOME dataset
            self.dome_distractors.set_visible(True)

            # Switch the distractors to DOME
            self.current_distractors = self.dome_distractors

            # Randomize the dome backgrounds
            self._setup_dome_randomizers()

            # Run another preview to generate the replicator graphs for the DOME dataset without triggering any writing
            rep.orchestrator.preview()

        # Randomize the distractors by applying forces to them and changing their materials
        self.current_distractors.apply_force_to_assets(config_data["FORCE_RANGE"])
        self.current_distractors.randomize_asset_glass_color()

        # Randomize the pose of the object(s) of interest in the camera view
        for train_part in self.train_parts:
            self.randomize_movement_in_view(train_part)

        # Simulate the applied forces for a couple of frames
        for _ in range(50):
            world.step(render=False)

        print(f"ID: {self.cur_idx}/{self.train_size - 1}")
        rep.orchestrator.step(rt_subframes=4)

        # Check that there was valid training data in the last frame (target object(s) visible to camera)
        if self.writer.is_last_frame_valid():
            self.cur_idx += 1

        # Check if last frame has been reached
        if self.cur_idx >= self.train_size:
            print(f"Dataset of size {self.train_size} has been reached, generation loop will be stopped..")
            print(f"Data outputted to: {self._output_folder}")
            self.last_frame_reached = True


dataset = RandomScenario(
    num_mesh=args.num_mesh,
    num_dome=args.num_dome,
    dome_interval=args.dome_interval,
    output_folder=args.output_folder,
    use_s3=args.use_s3,
    bucket=args.bucket,
    s3_region=args.s3_region,
    endpoint=args.endpoint,
    writer=args.writer.lower(),
    test=args.test,
)

if dataset.result:
    # Iterate through dataset and visualize the output
    print("Loading materials. Will generate data soon...")

    import datetime

    start_time = datetime.datetime.now()
    print("Start timestamp:", start_time.strftime("%m/%d/%Y, %H:%M:%S"))

    if dataset.train_size > 0:
        print(f"Starting dataset generation of {dataset.train_size} frames..")

        if dataset.num_mesh > 0:
            print(f"Starting MESH dataset generation of {dataset.num_mesh} frames..")

        # Dataset generation loop
        for _ in dataset:
            if dataset.last_frame_reached:
                print(f"Stopping generation loop at index {dataset.cur_idx}..")
                break
            if dataset.exiting:
                break
    else:
        print(f"Dataset size is set to 0 (num_mesh={dataset.num_mesh} num_dope={dataset.num_dome}), nothing to write..")

    print("End timestamp:", datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    print("Total time taken:", str(datetime.datetime.now() - start_time).split(".")[0])

if args.test:
    run_pose_generation_test(
        writer=args.writer,
        output_folder=dataset._output_folder,
        test_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"),
    )

# Close the app
kit.close()
