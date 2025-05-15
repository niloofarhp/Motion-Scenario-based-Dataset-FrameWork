# Copyright 2022 The Kubric Authors
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
Worker file for the Multi-Object Video (MOVi) C (and CC) datasets.
  * The number of objects is randomly chosen between
    --min_num_objects (3) and --max_num_objects (10)
  * The objects are randomly chosen from the Google Scanned Objects dataset

  * Background is an random HDRI from the HDRI Haven dataset,
    projected onto a Dome (half-sphere).
    The HDRI is also used for lighting the scene.
"""

# JL
import sys
import bpy
import numpy as np
import kubric as kb
import pybullet as pb
from kubric.renderer import Blender
from kubric.simulator import PyBullet

# import time
try:
    from scenes.movi_render import render
except:
    from movi_render import render

# The blender, pybullet and kubric scene description cannot be cleanly closed and
# restarted. Hence we use global variables to store the handles to them and try
# to reset them as needed.
simulator = None
renderer = None
scene = None
DEBUG_MOVI = False


def main(argv):
    # --- CLI arguments
    parser = kb.ArgumentParser()
    parser.add_argument("--objects_split", choices=["train", "test"],
                        default="train")
    # Configuration for the objects of the scene
    parser.add_argument("--min_num_objects", type=int, default=3,
                        help="minimum number of objects")
    parser.add_argument("--max_num_objects", type=int, default=10,
                        help="maximum number of objects")
    parser.add_argument("--motion", choices=["static", "slide", "rotate"],
                        default="rotate")
    # Will only affect slide mode
    parser.add_argument("--min_velocity", type=float, default=2.5)
    parser.add_argument("--max_velocity", type=float, default=5.0)
    # Will only affect rotate mode
    parser.add_argument("--min_rotational_velocity", type=float, default=2.5)
    parser.add_argument("--max_rotational_velocity", type=float, default=5.0)

    # Configuration for the floor and background
    parser.add_argument("--floor_friction", type=float, default=0.3)
    parser.add_argument("--floor_restitution", type=float, default=0.5)
    parser.add_argument("--backgrounds_split", choices=["train", "test"],
                        default="train")

    # Configuration of camera
    parser.add_argument("--camera", choices=["fixed_random", "linear_movement", "rotation", "path"],
                        default="fixed_random")
    parser.add_argument("--focal_len", type=float, default=35.)
    parser.add_argument("--cam_path", type=str, default="cam_path/path.txt")
    parser.add_argument("--max_camera_movement", type=float, default=4.0)
    parser.add_argument("--camera_max_num_circles", type=float, default=0.5)
    parser.add_argument("--camera_min_num_circles", type=float, default=0.5)
    parser.add_argument("--blocker", choices=["none", "texture"], default="none")
    parser.add_argument("--blocker_tpath", type=str, default="cam_textures/blocker.png")
    parser.add_argument("--fisheye", type=float, default=0.0)
    parser.add_argument("--rain", choices=["none", "texture", "proc"], default="none")  # "proc" for procedural rain
    parser.add_argument("--raindrop_num", type=int, default=300)  # number of raindrops
    parser.add_argument("--rain_area_size", type=float, default=3.0)  # size of procedural raindrops generation area
    parser.add_argument("--texture_fac", type=float, default=0.8)  # rain texture transparency
    parser.add_argument("--texture_path", type=str, default="cam_textures/raindrop.png")

    # Configuration for the source of the assets
    parser.add_argument("--assets", choices=["none", "fbx"], default="none")
    parser.add_argument("--kubasic_assets", type=str,
                        default="gs://kubric-public/assets/KuBasic/KuBasic.json")
    parser.add_argument("--hdri_assets", type=str,
                        default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    parser.add_argument("--gso_assets", type=str,
                        default="gs://kubric-public/assets/GSO/GSO.json")
    parser.add_argument("--fbx_path", type=str, default="fbx_models/walking.fbx")
    parser.add_argument("--save_state", dest="save_state", action="store_true")
    parser.set_defaults(save_state=False, frame_end=24, frame_rate=12, resolution=256)

    # Output directory
    parser.add_argument("--sub_dir", type=str, default="0001")

    FLAGS = parser.parse_args(argv[1:])
    if FLAGS.min_num_objects > FLAGS.max_num_objects:
        FLAGS.max_num_objects = FLAGS.min_num_objects

    job_dir_suffix = ''

    if FLAGS.assets != "none":
        job_dir_suffix += FLAGS.assets
        if FLAGS.blocker != "none":
            job_dir_suffix += '_' + FLAGS.blocker
    else:
        job_dir_suffix += FLAGS.motion
        if FLAGS.blocker != "none":
            job_dir_suffix += '_bar'
        job_dir_suffix += '_' + '{:03d}'.format(FLAGS.min_num_objects)

    FLAGS.job_dir = FLAGS.job_dir + '/' + FLAGS.camera + '_' + job_dir_suffix + '/' + FLAGS.sub_dir

    if FLAGS.fisheye > 0:
        FLAGS.job_dir = FLAGS.job_dir + '_fisheye'
    if FLAGS.rain != 'none':
        FLAGS.job_dir = FLAGS.job_dir + '_env'

    print(f'\nFLAGS: {FLAGS}')
    if DEBUG_MOVI:
        if 'simulator' in globals():
            print("Simulator in globals.")
            if simulator is not None:
                print("Simulator not None.")
            else:
                print("Simulator None")
        else:
            print("Simulator missing")
    renderSetup(FLAGS)


# Modified kb utils to start scene description with existing scene object
def setupWithScene(flags, scene):
    kb.setup_logging(flags.logging_level)
    kb.log_my_flags(flags)

    seed = flags.seed if flags.seed else np.random.randint(0, 2147483647)
    rng = np.random.RandomState(seed=seed)
    scene.metadata["seed"] = seed

    scratch_dir, output_dir = kb.setup_directories(flags)
    return rng, output_dir, scratch_dir


# Code to inquire pyBullet connection status
# try:
#   sim_id = pb.connect(pb.DIRECT)
#   print( pb.getConnectionInfo(sim_id) )
#   pb.disconnect(sim_id)
# except:
#   print("pyBullet: Could not connect and disconnect.")

# Test if pyBullet client simulator accepts addition and queries
# print( "Simulator id: {0}".format(simulator.physics_client) )
# mystery = pb.addUserDebugLine([0.1,0.2,0.3],[0.5,0.7,0.7])
# print(mystery)
# pb.getBaseVelocity(mystery)

# movie rendering start and tear down
def renderSetup(FLAGS):
    global DEBUG_MOVI
    # --- Common setups & resources
    print("\nkb.setup ... ")
    global scene
    global renderer
    global simulator
    # See what's in blender - code from blender manual
    if renderer is not None:
        # print all objects
        for obj in bpy.data.objects:
            print(obj.name)
            # print all scene names in a list
            print(bpy.data.scenes.keys())
        # print all meshes
        for mesh in bpy.data.meshes:
            print(mesh.name)

    if scene is not None:
        if DEBUG_MOVI:
            print("Scene not none.")
        rng, output_dir, scratch_dir = setupWithScene(FLAGS, scene)
        for asset in scene._assets:
            scene.remove(asset)
            if renderer is not None:
                renderer.remove_asset(asset)
            if simulator is not None:
                simulator.remove_asset(asset)
        # unlinking views corrupts references
        # for view in scene._views:
        #   scene.unlink_view(view)
        # force assets in scene description to empty
        scene._assets = []
    # scene._views = []
    # scene.metadata = {}
    else:
        if DEBUG_MOVI:
            print("Creating Scene")
        scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    if DEBUG_MOVI:
        print("... done!")

    if renderer is not None:
        # print and remove all objects
        print("All assets should be removed: ")
        for obj in bpy.data.objects:
            print(obj.name)
        # if obj.name != 'PerspectiveCamera':
        #     print("Removing obj: ", obj.name)
        #     bpy.data.objects.remove(obj)
        # print and remove all meshes
        for mesh in bpy.data.meshes:
            print("Removing mesh: ", mesh.name)
            bpy.data.meshes.remove(mesh)
        for mat in bpy.data.materials:
            print("Removing material: ", mat.name)
            bpy.data.materials.remove(mat)
        for text in bpy.data.textures:
            print("Removing texture: ", text.name)
            bpy.data.textures.remove(text)
        # print all scene names in a list
        print("Scenes in Blender: ", bpy.data.scenes.keys())
    # Removing cameras corrupts Blender references
    # ReferenceError: StructRNA of type Scene has been removed
    # try:
    #     bpy.data.objects['PerspectiveCamera'].select_set(True) # Blender 2.8x
    #     bpy.ops.object.delete()
    #     print("Blender: Removed perspective camera")
    # except:
    #     print("Blender: No perspective camera to remove")

    if simulator is not None:
        if DEBUG_MOVI:
            print("Simulator not none.")
        pb.resetSimulation(simulator.physics_client)
        simulator.scratch_dir = scratch_dir
        if scene is not None:
            simulator.scene = scene
    else:
        if DEBUG_MOVI:
            print("Creating Simulator")
        simulator = PyBullet(scene, scratch_dir)

    if renderer is not None:
        if DEBUG_MOVI:
            print("Renderer not none.")
        # renderer.clear_and_reset_blender_scene(True) # ReferenceError: StructRNA of type Scene has been removed
        renderer.scratch_dir = scratch_dir
        if scene is not None:
            renderer.scene = scene
    else:
        if DEBUG_MOVI:
            print("Creating Renderer")
        renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
        renderer.use_gpu = True

    render(FLAGS, scene, renderer, simulator, output_dir, rng)

    # print( "Simulator id: {0}".format(simulator.physics_client) )
    # try:
    #   pb.resetSimulation(simulator.physics_client)
    #   pb.disconnect(simulator.physics_client)
    #   print("pyBullet: Disconnect worker successful")
    # except:
    #   print("pyBullet: Failure to disconnect worker")

    # kb.done()
    # from kubric import assets  # pylint: disable=import-outside-toplevel
    # assets.ClosableResource.close_all()
    import subprocess
    subprocess.run(["ffmpeg", "-i", "{0}/rgba_%05d.png".format(output_dir), "{0}/video.mp4".format(output_dir)])


if __name__ == '__main__':
    main(sys.argv)
