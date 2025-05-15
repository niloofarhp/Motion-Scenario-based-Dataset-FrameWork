# Copyright 2022 Jochen Lang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Worker file for generation of movies in which an objects translates 
across the scene
  * The file is based on MOVI_C
  * The number of objects can be randomly chosen 
    --min_num_objects (3) and --max_num_objects (10)
  * The objects are randomly chosen from the Google Scanned Objects dataset
  * Background is an random HDRI from the HDRI Haven dataset,
    projected onto a Dome (half-sphere).
    The HDRI is also used for lighting the scene.
"""

import os
import sys
import math
import random
import logging
import contextlib

import bpy
import numpy as np
import kubric as kb
import pybullet as pb
import rain_generator as rain

import scipy.interpolate as si

# --- Some configuration values
# the region in which to place objects [(min), (max)]
# reduce region to central area close to ground
SPAWN_REGION = [(-1, -1, 0.5), (1, 1, 1.5)]
# SPAWN_REGION = [(-3, -3, 1.5), (3, 3, 5.5)]
VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]


@contextlib.contextmanager
def suppress_output():
    """Suppress the warnings about unsupported user property types in Blender"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_linear_camera_motion_start_end(
        rng,
        movement_speed: float,
        inner_radius: float = 8.,
        outer_radius: float = 12.,
        z_offset: float = 0.1,
):
    """Sample a linear path which starts and ends within a half-sphere shell."""
    while True:
        camera_start = np.array(
            kb.sample_point_in_half_sphere_shell(
                inner_radius, outer_radius, z_offset))
        direction = rng.rand(3) - 0.5
        movement = direction / np.linalg.norm(direction) * movement_speed
        camera_end = camera_start + movement
        if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
                camera_end[2] > z_offset):
            return camera_start, camera_end


def get_camera_in_spherical_coordinates(
        inner_radius: float = 8.,
        outer_radius: float = 12.,
        z_offset: float = 0.1,
):
    original_camera_position = tuple(
        kb.sample_point_in_half_sphere_shell(
            inner_radius, outer_radius, z_offset))

    return pos_cartesian_to_spherical(original_camera_position)


def pos_cartesian_to_spherical(
        pos_xyz: tuple
):
    # phi is elevation measured from z
    # theta is azimuth
    r = np.sqrt(sum(a * a for a in pos_xyz))
    theta = np.arctan2(pos_xyz[1], pos_xyz[0])
    phi = np.arctan2(math.sqrt(pos_xyz[0] ** 2 + pos_xyz[1] ** 2), pos_xyz[2])
    return r, phi, theta


def pos_spherical_to_cartesian(
        pos_r_phi_theta: tuple
):
    x = pos_r_phi_theta[0] * np.cos(pos_r_phi_theta[2]
                                    ) * np.sin(pos_r_phi_theta[1])
    y = pos_r_phi_theta[0] * np.sin(pos_r_phi_theta[2]
                                    ) * np.sin(pos_r_phi_theta[1])
    z = pos_r_phi_theta[0] * np.cos(pos_r_phi_theta[1])
    return x, y, z


def read_camera_path(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file does not exist: {file_path}")
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.strip().split())
            points.append((x, y, z))
    return points


# TODO: Kubric blender.py assertion error
# def wrap_blender_to_kubric(blender_obj, fbx_path):
#     """
#     Wrap a blender object and its children into Kubric assets
#     """
#
#     def get_bounds(obj):
#         min_x, min_y, min_z = obj.bound_box[0]
#         max_x, max_y, max_z = obj.bound_box[0]
#
#         for vertex in obj.bound_box:
#             min_x = min(min_x, vertex[0])
#             min_y = min(min_y, vertex[1])
#             min_z = min(min_z, vertex[2])
#             max_x = max(max_x, vertex[0])
#             max_y = max(max_y, vertex[1])
#             max_z = max(max_z, vertex[2])
#
#         return (min_x, min_y, min_z), (max_x, max_y, max_z)
#
#     def wrap_object(obj, segmentation_id):
#         bounds = get_bounds(obj)
#         return kb.FileBasedObject(
#             asset_id=obj.name,
#             segmentation_id=segmentation_id,
#             render_filename=fbx_path,
#             bounds=bounds,
#             simulation_filename=None
#         )
#
#     def process_children(parent_obj, segmentation_id):
#         k_objs = [wrap_object(parent_obj, segmentation_id)]
#         for obj in parent_obj.children:
#             if obj.type == 'MESH':
#                 obj["segmentation_id"] = segmentation_id
#                 k_objs.append(wrap_object(obj, segmentation_id))
#                 segmentation_id += 1
#             elif obj.children:
#                 # Recursively process nested children
#                 nested_kubric_objs, segmentation_id = process_children(obj, segmentation_id)
#                 k_objs.extend(nested_kubric_objs)
#         return k_objs, segmentation_id
#
#     kubric_objs, _ = process_children(blender_obj, 1)
#     return kubric_objs


def import_fbx(fbx_path, scale_factor=0.025):
    armature_obj = None

    with suppress_output():
        bpy.ops.object.select_all(action='DESELECT')  # clear existing object selection
        bpy.ops.import_scene.fbx(filepath=fbx_path)
        all_objects = bpy.context.selected_objects
        bpy.ops.object.select_all(action='DESELECT')  # deselect all objects to ensure a clean state

    seg_id = 1
    for obj in all_objects:
        if obj.type == 'ARMATURE':
            armature_obj = obj
            obj['segmentation_id'] = seg_id
            seg_id += 1
        elif obj.type == 'MESH':
            # kubric_objs.extend(wrap_blender_to_kubric(obj, fbx_path))  # wrap the object and its children
            obj['segmentation_id'] = seg_id
            seg_id += 1
        else:
            obj.select_set(False)  # deselect non-mesh and non-armature objects

    if armature_obj is None:
        raise ValueError("No Armature found in the imported FBX file.")

    bpy.context.view_layer.objects.active = armature_obj  # set the armature as active
    armature_obj.scale = (scale_factor, scale_factor, scale_factor)  # apply scaling to armature

    logging.info("Objects from FBX file have been successfully loaded.")
    return all_objects  # Return all imported objects


def apply_lens_effects(FLAGS):
    scene = bpy.context.scene
    tree = scene.node_tree
    links = tree.links

    # environment texture node
    env_texture_node = tree.nodes.new(type='CompositorNodeImage')
    env_texture_node.location = (400, -250)
    if FLAGS.rain == 'texture':
        env_texture_node.image = bpy.data.images.load(FLAGS.texture_path)

    # transform node for env texture
    env_transform_node = tree.nodes.new(type='CompositorNodeTransform')
    env_transform_node.location = (600, -200)
    env_transform_node.inputs['Angle'].default_value = random.uniform(-10, 10)
    env_transform_node.inputs['Scale'].default_value = random.uniform(1, 2)

    # scale node for environment render size
    env_scale_node = tree.nodes.new(type='CompositorNodeScale')
    env_scale_node.location = (800, -200)
    env_scale_node.space = 'RENDER_SIZE'
    env_scale_node.frame_method = 'STRETCH'

    # blocker texture node
    blocker_node = tree.nodes.new(type='CompositorNodeImage')
    blocker_node.location = (400, -600)
    if FLAGS.blocker != 'none':
        blocker_node.image = bpy.data.images.load(FLAGS.blocker_tpath)

    # transform node for blocker texture
    blocker_transform_node = tree.nodes.new(type='CompositorNodeTransform')
    blocker_transform_node.location = (800, -500)
    # blocker_transform_node.inputs['Angle'].default_value = random.uniform(-10, 10)  # random rotation of the blocker
    blocker_transform_node.inputs['Scale'].default_value = random.uniform(1, 2)

    # scale node for blocker render size
    blocker_scale_node = tree.nodes.new(type='CompositorNodeScale')
    blocker_scale_node.location = (600, -500)
    blocker_scale_node.space = 'RENDER_SIZE'
    blocker_scale_node.frame_method = 'FIT'

    # alpha over node for env texture and render layer combination
    alpha_over_env_render = tree.nodes.new(type='CompositorNodeAlphaOver')
    alpha_over_env_render.location = (1000, -200)
    alpha_over_env_render.inputs['Fac'].default_value = FLAGS.texture_fac

    # alpha over node for blocker and combined env
    alpha_over_blocker_render = tree.nodes.new(type='CompositorNodeAlphaOver')
    alpha_over_blocker_render.location = (1000, -400)
    if FLAGS.blocker != 'none':
        alpha_over_blocker_render.inputs['Fac'].default_value = 1
    else:
        alpha_over_blocker_render.inputs['Fac'].default_value = 0

    # lens distortion node
    distort_node = tree.nodes.new(type='CompositorNodeLensdist')
    distort_node.location = (1200, -200)
    distort_node.inputs["Distort"].default_value = FLAGS.fisheye

    # get the render layers and output nodes from current tree
    render_layers = tree.nodes.get('Render Layers')  # render layer must exist

    # composite node
    composite_node = None
    for node in tree.nodes:
        if node.type == 'COMPOSITE':
            composite_node = node
            composite_node.location = (1400, -200)
            break

    # If no Composite node is found, create one
    if composite_node is None:
        composite_node = tree.nodes.new(type='CompositorNodeComposite')
        composite_node.location = (1400, -200)

    # links for env
    links.new(env_texture_node.outputs[0], env_transform_node.inputs[0])
    links.new(env_transform_node.outputs[0], env_scale_node.inputs[0])
    links.new(env_scale_node.outputs[0], alpha_over_env_render.inputs[2])

    # links for blocker
    links.new(blocker_node.outputs[0], blocker_scale_node.inputs[0])
    links.new(blocker_scale_node.outputs[0], blocker_transform_node.inputs[0])
    links.new(blocker_transform_node.outputs[0], alpha_over_blocker_render.inputs[2])

    # links for final blend
    links.new(render_layers.outputs[0], alpha_over_env_render.inputs[1])
    links.new(alpha_over_env_render.outputs[0], alpha_over_blocker_render.inputs[1])
    links.new(alpha_over_blocker_render.outputs[0], distort_node.inputs[0])
    links.new(distort_node.outputs[0], composite_node.inputs[0])


# Function to compute Bezier curve points
def bezier_curve(points, num_points=100):
    """
    Computes points on a Bezier curve defined by 'points'.
    :param points: A list of control points for the Bezier curve.
    :param num_points: Number of points to interpolate along the curve.
    :return: A list of interpolated points along the Bezier curve.
    """
    n = len(points) - 1
    combinations = np.array([math.comb(n, i) for i in range(n + 1)])
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 3))

    for i in range(n + 1):
        curve += np.outer(combinations[i] * (t ** i) * ((1 - t) ** (n - i)), points[i])

    return curve


def apply_camera_path_with_bezier(scene, camera_path, frame_start, frame_end):
    if len(camera_path) < 2:
        raise ValueError("The camera path file must contain at least two points.")

    # compute the Bezier curve points for the camera path
    bezier_points = bezier_curve(camera_path, num_points=(frame_end - frame_start + 3))

    for frame, position in enumerate(bezier_points, start=frame_start - 1):
        scene.camera.position = position
        scene.camera.look_at((0, 0, 0))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)

    # Ensure that the last frame is also set
    scene.camera.position = bezier_points[-1]
    scene.camera.keyframe_insert("position", frame_end + 1)
    scene.camera.keyframe_insert("quaternion", frame_end + 1)


# Main rendering loop
def render(FLAGS, scene, renderer, simulator, output_dir, rng):
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

    # --- Populate the scene
    # background HDRI
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    if FLAGS.backgrounds_split == "train":
        logging.info("Choosing one of the %d training backgrounds...",
                     len(train_backgrounds))
        hdri_id = rng.choice(train_backgrounds)
    else:
        logging.info("Choosing one of the %d held-out backgrounds...",
                     len(test_backgrounds))
        hdri_id = rng.choice(test_backgrounds)
    background_hdri = hdri_source.create(asset_id=hdri_id)
    # assert isinstance(background_hdri, kb.Texture)
    logging.info("Using background %s", hdri_id)
    scene.metadata["background"] = hdri_id
    renderer._set_ambient_light_hdri(background_hdri.filename)

    # Dome
    dome = kubasic.create(asset_id="dome", name="dome",
                          friction=FLAGS.floor_friction,
                          restitution=FLAGS.floor_restitution,
                          static=True, background=True)
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome.friction = 1.0  # reduce sliding and rotation
    dome.restitution = 0.0  # no bounce
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)

    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=FLAGS.focal_len, sensor_width=32)

    if FLAGS.camera == "fixed_random":
        scene.camera.position = kb.sample_point_in_half_sphere_shell(
            inner_radius=10., outer_radius=9., offset=0.1)
        scene.camera.look_at((0, 0, 0))

    elif FLAGS.camera == "linear_movement":
        camera_start, camera_end = get_linear_camera_motion_start_end(
            rng, movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement))
        # linearly interpolate the camera position between these two points
        # but keeping the camera in the same orientation
        # we start one frame early and end one frame late to ensure that
        # forward and backward flow are still consistent for the last and first frames

        # solve for camera orientation
        # interpolate position for middle position
        middle_pos = 0.5 * (np.array(camera_end) + np.array(camera_start))
        scene.camera.position = middle_pos
        scene.camera.look_at((0, 0, 0))
        middle_quaternion = scene.camera.quaternion

        for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
            interp = ((frame - FLAGS.frame_start + 1) /
                      (FLAGS.frame_end - FLAGS.frame_start + 3))
            # Interpolation formula corrected JL
            scene.camera.position = (interp * np.array(camera_end) +
                                     (1 - interp) * np.array(camera_start))
            # scene.camera.look_at((0, 0, 0))
            scene.camera.quaternion = middle_quaternion
            scene.camera.keyframe_insert("position", frame)
            scene.camera.keyframe_insert("quaternion", frame)

    elif FLAGS.camera == "mixed_movement":
        camera_start, camera_end = get_linear_camera_motion_start_end(
            rng, movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement))
        # linearly interpolate the camera position between these two points
        # while keeping it focused on the center of the scene
        # we start one frame early and end one frame late to ensure that
        # forward and backward flow are still consistent for the last and first frames

        for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
            interp = ((frame - FLAGS.frame_start + 1) /
                      (FLAGS.frame_end - FLAGS.frame_start + 3))
            # Interpolation formula corrected JL
            scene.camera.position = (interp * np.array(camera_end) +
                                     (1 - interp) * np.array(camera_start))
            scene.camera.look_at((0, 0, 0))
            scene.camera.keyframe_insert("position", frame)
            scene.camera.keyframe_insert("quaternion", frame)

    elif FLAGS.camera == "rotation":
        # --- Keyframe a circular camera path around the object (use polar coordinates)
        # Adapted from example keyframing
        r, phi, theta = get_camera_in_spherical_coordinates()

        theta_change = (2 * np.pi) / (scene.frame_end - scene.frame_start) * rng.uniform(
            low=FLAGS.camera_min_num_circles, high=FLAGS.camera_max_num_circles)
        frame_extra = 0

        for frame in range(scene.frame_start - frame_extra, scene.frame_end + 1 + frame_extra):
            i = (frame - scene.frame_start + frame_extra)
            theta_new = i * theta_change + theta

            # These values of (x, y, z) will lie on the same sphere as the original camera.
            # x = r * np.cos(theta_new) * np.sin(phi)
            # y = r * np.sin(theta_new) * np.sin(phi)
            # z = r * np.cos(phi)
            x, y, z = pos_spherical_to_cartesian((r, phi, theta_new))

            if frame >= scene.frame_start and frame <= scene.frame_end + 1:
                scene.camera.position = (x, y, z)
                scene.camera.look_at((0, 0, 0))
                scene.camera.keyframe_insert("position", frame)
                scene.camera.keyframe_insert("quaternion", frame)

    elif FLAGS.camera == "path":
        camera_path = read_camera_path(FLAGS.cam_path)  # read camera path from file
        apply_camera_path_with_bezier(scene, camera_path, FLAGS.frame_start, FLAGS.frame_end)

    # apply the lens effects if any lens effects related flags
    if FLAGS.blocker != 'none' or FLAGS.rain != 'none' or FLAGS.fisheye >= 0:
        if FLAGS.rain == 'proc':
            logging.info("Generating procedural raindrops...")
            rain.add_raindrops_to_camera(scene.camera.linked_objects[renderer],
                                         FLAGS.raindrop_num, FLAGS.rain_area_size)
        apply_lens_effects(FLAGS)

    if FLAGS.assets == "fbx":
        num_objects = 0  # TODO: skipping pybullet for custom model to prevent errors for now
        active_split = []
    else:
        # Add random objects
        train_split, test_split = gso.get_test_split(fraction=0.1)
        if FLAGS.objects_split == "train":
            logging.info("Choosing one of the %d training objects...",
                         len(train_split))
            active_split = train_split
        else:
            logging.info("Choosing one of the %d held-out objects...",
                         len(test_split))
            active_split = test_split

        num_objects = rng.randint(FLAGS.min_num_objects, FLAGS.max_num_objects + 1)
        logging.info("Randomly placing %d objects:", num_objects)

    # create random velocities in the x/y plane for all objects
    obj_vel = []
    obj_angular_vel = []
    obj_spawn = []

    if FLAGS.motion == "static":
        logging.info("FLAGS.motion = static")
        for i in range(num_objects):
            obj_vel.append((0, 0, 0))
            obj_spawn.append(SPAWN_REGION)
            obj_angular_vel.append((0, 0, 0))

    elif FLAGS.motion == "slide":
        logging.info("FLAGS.motion = slide")
        for i in range(num_objects):
            vel_mag = rng.uniform(FLAGS.min_velocity, FLAGS.max_velocity)
            vel_dir = rng.uniform(-math.pi, math.pi)
            vel = (vel_mag * math.cos(vel_dir),
                   vel_mag * math.sin(vel_dir), 0.0)
            obj_vel.append(vel)
            sp = [(x - vel[0], y - vel[1], z) for (x, y, z) in SPAWN_REGION]
            obj_spawn.append(sp)
            obj_angular_vel.append((0, 0, 0))

    elif FLAGS.motion == "rotate":
        logging.info("FLAGS.motion = rotate")
        for i in range(num_objects):
            vel_rot = rng.uniform(
                FLAGS.min_rotational_velocity, FLAGS.max_rotational_velocity)
            obj_vel.append((0, 0, 0))
            obj_spawn.append(SPAWN_REGION)
            obj_angular_vel.append((0, 0, vel_rot))

    # separate creating objects from adding them to the scene
    obj_list = []

    # Pybullet objects setups
    if FLAGS.assets == "fbx":
        logging.info(f"Loading custom fbx model from {FLAGS.fbx_path}")
        if not FLAGS.fbx_path.endswith('.fbx'):
            raise ValueError(f"Unsupported file format for {FLAGS.fbx_path}. Please provide a valid '.fbx' file")

        try:
            imported_objects = import_fbx(FLAGS.fbx_path)  # Directly import the FBX file into the blender scene
            # for obj in imported_objects:
            #     # Deselect all objects to ensure a clean state
            #     bpy.ops.object.select_all(action='DESELECT')
            #     blender_obj = bpy.data.objects.get(obj.asset_id)
            #     blender_obj.select_set(True)
            #     bpy.context.view_layer.objects.active = blender_obj
            #
            #     logging.info(f"Selected objects{len(bpy.context.selected_objects)}: {bpy.context.selected_objects}")
            #
            #     # Add the object to the scene
            #     logging.info(f"Adding object {obj.asset_id} to the scene.")
            #     scene.add(obj)  # Adding the wrapped object to the scene
        except Exception as e:
            logging.error(f"Failed to load the custom FBX model: {e}")
            raise

    # Pybullet compatible objects
    else:
        for _ in range(num_objects):
            obj = gso.create(asset_id=rng.choice(active_split))
            assert isinstance(obj, kb.FileBasedObject)  # assert object is kb.FileBasedObject
            scale = rng.uniform(1.5, 3.0)  # No tiny objects
            obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
            obj.friction = 1.0
            obj.restitution = 0.0
            obj.metadata["scale"] = scale
            obj_list.append(obj)

        logging.info("PyBullet connection info: " + str(pb.getConnectionInfo(simulator.physics_client)))
        for i in range(num_objects):
            scene += obj_list[i]
            attempts = 20
            while attempts > 0:
                try:
                    kb.move_until_no_overlap(
                        obj_list[i], simulator, spawn_region=obj_spawn[i], rng=rng)
                    logging.info("    Added %s at %s",
                                 obj_list[i].asset_id, obj_list[i].position)
                    break
                except:
                    # Increase spawn region and try again
                    attempts -= 1
                    logging.error("Failed to add object in attempt {0}".format(10 - attempts))
                    inc_step = 0.5
                    lower_left = obj_spawn[i][0]
                    lower_left = (lower_left[0] - inc_step,
                                  lower_left[1] - inc_step, lower_left[2])
                    upper_right = obj_spawn[i][1]
                    upper_right = (
                        upper_right[0] + inc_step, upper_right[1] + inc_step,
                        upper_right[2] + 2 * inc_step)
                    obj_spawn[i] = [lower_left, upper_right]

    logging.info("Running 200 frames of simulation to let static objects settle ...")
    _, _ = simulator.run(frame_start=-200, frame_end=0)

    # for obj in scene.foreground_assets:
    for i in range(num_objects):
        obj = obj_list[i]
        if hasattr(obj, "position"):
            logging.info(obj.position)
        if hasattr(obj, "velocity"):
            logging.info(obj.velocity)
            # z is up, x is right, y is in
            obj.velocity = obj_vel[i]
            obj.friction = 0.0  # set velocity to small number and reset friction / restitution
            obj.restitution = 0.5
        if hasattr(obj, "angular_velocity"):
            logging.info(obj.angular_velocity)
            obj.angular_velocity = obj_angular_vel[i]

    # reset floor to defined properties
    dome.friction = FLAGS.floor_friction
    dome.restitution = FLAGS.floor_restitution

    if FLAGS.save_state:
        logging.info("Saving the simulator state to '%s'", output_dir / "scene.bullet")
        simulator.save_state(output_dir / "scene.bullet")
        renderer.save_state(output_dir / "scene.blend")

    # run dynamic objects simulation
    logging.info("\nRunning the simulation ...")
    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end + 1)
    logging.info("Rendering the scene (with %s engine) ...", bpy.context.scene.render.engine)

    data_stack = renderer.render()
    print()

    # --- Postprocessing

    # Check kubric scene's segmentation map:
    # segmentation_map = data_stack["segmentation"]
    # unique_values = np.unique(segmentation_map)
    # logging.info(f"Segmentation map unique values: {unique_values}")

    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                 if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True)

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"],
        scene.assets,
        visible_foreground_assets)
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    # save image files
    kb.write_image_dict(data_stack, output_dir)
    kb.post_processing.compute_bboxes(data_stack["segmentation"], visible_foreground_assets)

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    kb.write_json(filename=output_dir / "metadata.json", data={
        "flags": vars(FLAGS),
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "instances": kb.get_instance_info(scene, visible_foreground_assets),
    })
    kb.write_json(filename=output_dir / "events.json", data={
        "collisions": kb.process_collisions(
            collisions, scene, assets_subset=visible_foreground_assets),
    })
