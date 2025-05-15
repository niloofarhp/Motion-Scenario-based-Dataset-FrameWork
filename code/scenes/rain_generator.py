from mathutils import Vector
import numpy as np
import bpy


def generate_raindrops(num_drops, area_size):
    """Generate a list of raindrop locations and sizes."""
    drops = []
    for _ in range(num_drops):
        x = np.random.uniform(-area_size, area_size)
        y = np.random.uniform(-area_size, area_size)
        size = np.random.uniform(0.01, 0.15)  # adjust single raindrops size
        drops.append((x, y, size))
    return drops


def create_raindrop_material():
    """Create a realistic raindrop material with randomized shapes using nodes."""
    mat = bpy.data.materials.new(name="RaindropMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add necessary nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    transparent_node = nodes.new(type='ShaderNodeBsdfTransparent')
    mix_shader_node = nodes.new(type='ShaderNodeMixShader')
    fresnel_node = nodes.new(type='ShaderNodeFresnel')
    noise_texture_node = nodes.new(type='ShaderNodeTexNoise')
    voronoi_texture_node = nodes.new(type='ShaderNodeTexVoronoi')
    color_ramp_node = nodes.new(type='ShaderNodeValToRGB')
    bump_node = nodes.new(type='ShaderNodeBump')

    # Configure nodes
    principled_node.inputs['Transmission'].default_value = 1.0
    principled_node.inputs['Roughness'].default_value = 0.1
    principled_node.inputs['IOR'].default_value = 1.33

    noise_texture_node.inputs['Scale'].default_value = 50.0
    voronoi_texture_node.inputs['Scale'].default_value = 70.0

    color_ramp_node.color_ramp.interpolation = 'EASE'
    color_ramp_node.color_ramp.elements[0].position = 0.4
    color_ramp_node.color_ramp.elements[1].position = 0.6

    bump_node.inputs['Strength'].default_value = 0.1

    # Connect nodes
    links.new(output_node.inputs['Surface'], mix_shader_node.outputs[0])
    links.new(mix_shader_node.inputs[1], transparent_node.outputs[0])
    links.new(mix_shader_node.inputs[2], principled_node.outputs[0])
    links.new(mix_shader_node.inputs['Fac'], fresnel_node.outputs['Fac'])

    links.new(fresnel_node.inputs['IOR'], noise_texture_node.outputs['Fac'])
    links.new(noise_texture_node.outputs['Fac'], color_ramp_node.outputs['Color'])
    links.new(color_ramp_node.outputs['Color'], mix_shader_node.inputs['Fac'])

    links.new(voronoi_texture_node.outputs['Distance'], bump_node.inputs['Height'])
    links.new(bump_node.outputs['Normal'], principled_node.inputs['Normal'])

    return mat


def add_raindrops_to_camera(camera, num_drops=300, area_size=3.0):
    """Add raindrops to the camera as sphere objects."""
    drops = generate_raindrops(num_drops, area_size)
    raindrop_material = create_raindrop_material()

    for drop in drops:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=drop[2] / 2, location=(0, 0, 0))
        sphere = bpy.context.object
        sphere.name = "Raindrop"
        sphere.parent = camera
        start_location = Vector((drop[0], drop[1], -2))
        end_location = Vector((drop[0], drop[1] - .5, -2))  # Move down to -1.0 for example

        sphere.location = start_location
        sphere.rotation_euler = camera.rotation_euler  # Align the sphere to face the camera
        sphere.data.materials.append(raindrop_material)

        # Insert keyframe at frame 0
        sphere.location = start_location
        sphere.keyframe_insert(data_path="location", frame=0)

        # Insert keyframe at frame 24
        sphere.location = end_location
        sphere.keyframe_insert(data_path="location", frame=24)
