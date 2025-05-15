# FlowDataset

## Description

This is a kubric based data generation project. The purpose is to be able to systematically generate different videos
with motion for estimating optical flow.

The videos are categorized based on object motion, camera motion, clutter and the presence of a camera-centered blocker.

## Visuals

ToDo: Add some example results.

## Installation

This project is designed to run in a Docker environment on an Ubuntu sub-system. Follow these steps for installation:

1. **Install Docker**: Download and install Docker from the official website: [Docker](https://www.docker.com/).

2. **Windows Users**:
    - Install the Windows Subsystem for Linux (WSL) by following the guide
      here: [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install).
    - It is recommended to install the Windows Terminal app for a better experience.

3. **Recommended Editor**: For editing and managing your project, we recommend using Visual Studio Code, which can be
   downloaded from [here](https://code.visualstudio.com/).

After completing these steps, your environment should be set up and ready for use.

## Usage Guide

### Prerequisites

1. **Hydra Integration**: We use Hydra for managing multiple runs of the base code. Configuration options are specified
   in the `conf/` subdirectory.
2. **Video Generator**: The core functionality is defined in `scenes/movi.py` and `scenes/movi_render.py`,
   with `parseworker.py` as the top-level program.
3. **Execution Options**: Two distinct usage options are available, listed in `generateDocker.sh`.

### Recommended Steps for Setup

1. **Set Working Directory**:
   Navigate to the installation directory of FlowDataset.

```bash
cd <install_directory> # Replace <install_directory> with the actual path.
```

2. **Docker Image Creation**:
   Execute the `generateDocker.sh` script to build a Kubric docker image equipped with Hydra and FFmpeg:

```bash
./generateDocker.sh
```

- This step creates a docker image named `local/hydra_kubric`.

3. **Basic Usage Test**:
   Run the following Docker command to test the basic functionality:

```bash
docker run --gpus all --rm -i -u $(id -u):$(id -g) -v "$(pwd):/workspace" local/hydra_kubric python3 parseworker.py
```

- This generates a video in the `output` subdirectory. The parameters prefixed with `--` are Docker options, crucial
  among them being the shared volume for running Python scripts and accessing output.

4. **Customizing Parameters for Video Generation**:
   To try different parameters for the movie generator, use:

```bash
docker run --gpus all --rm -i -u $(id -u):$(id -g) -v "$(pwd):/workspace" local/hydra_kubric python3 parseworker.py motion='slide' camera='fixed_random'
```

This command passes parameters for object motion (`motion`) and camera movement (`camera`) to the generator.

5. **Configuration Options**:
   Default options can be found in the `conf` directory. Hydra employs a hierarchical structure for configuration.

6. **Generating Multiple Videos**:
   For running multiple video generations, execute:

```bash
docker run --gpus all --rm -i -u $(id -u):$(id -g) -v "$(pwd):/workspace" local/hydra_kubric python3 parseworker.py -m
```

This runs all the combinations set under `conf/movement/rotate/all` as shown in the setting in `conf/config.yaml` file

```bash
docker run --gpus all --rm -i -u $(id -u):$(id -g) -v "$(pwd):/workspace" local/hydra_kubric python3 parseworker.py -m movement='slide/all'
```

This runs all the combinations set under `conf/movement/slide/all.yaml`

The `-m` option let Hydra use all combinations specified under `# hydra.mode --multirun` in the configuration files.
Modify `conf/config.yaml`, `conf/movement/static/all.yaml`, `conf/movement/rotate/all.yaml`
and `conf/movement/slide/all.yaml` to define the set of runs.


### Adjusting Fisheye Effect and Environmental Effect Strength

You can adjust the strength of the fisheye and environmental effects (e.g., raindrop by default) using Hydra flags 
`fisheye` and `rain`. The `fisheye` flag accepts values ranging from 0 to 1. The `rain` flag has three choices: 
`none`, `texture`, and `proc`. The `texture` option applies a texture of your choice (specified by `texture_path`) 
on the camera, while the `proc` option uses procedural rain. Procedural rain can be configured with two flags: 
`rain_area_size` and `raindrop_num`. Through `raindrop_num`, you can control how many raindrops are generated 
within the square area defined by `rain_area_size` (in meters).

**Fisheye Distortion in ground truth flow**
   - Controlled by the `FISHEYE_DISTORTION` constant in flow_dataset.py (default = 0.05)
   - Applies a barrel distortion effect to the optical flow
   - Values between 0.0 (no distortion) and 1.0 (maximum distortion)

**Using Custom Blocker Image for ground truth flow**

In flow_dataset.py:
- Loads a custom PNG image to define blocker regions
- The image will be automatically:
    - Resized to match flow dimensions (256x256)
    - Converted to grayscale if colored
    - Thresholded to create binary mask (pixels > 128 become blocker regions)

```python
# In flow_dataset.py, the blocker is applied with:
masked_flow = apply_blocker_mask(
    optical_flow=distorted_flow,
    blocker_image_path="../cam_textures/blocker.png"
)
```


**Example Usage**

To set the transparency of the environmental texture to 50% and a standard fisheye camera, use the following command:

```bash
docker run --gpus all --rm -i -u $(id -u):$(id -g) -v "$(pwd):/workspace" local/hydra_kubric python3 parseworker.py rain='texture' texture_fac=0.5 fisheye=1
```

To generate procedural rain with 250 rain drops in 2x2 meters in front of camera (docker commands omitted):

```bash
... python3 parseworker.py rain='proc' fisheye=1 rain_area_size=2 raindrop_num=250
```

To use a custom environment or blocker texture, specify the path to your file using the texture_path or blocker_tpath parameter:

bash


```bash
texture_path='your/path/to/texture.png'
```

or

```bash
blocker_tpath='your/path/to/blocker.png'
```


### Importing Custom Animated FBX Model

This guide will assist you in setting up custom animated model for video generation.

1. Create a folder named `fbx_models` in the main directory if it doesn't already exist.
   Place your .fbx model in this folder.

    - By default, a walking character model [walking.fbx](fbx_models/walking.fbx) is provided in the folder
      for testing purposes.

2. Testing with the Default Model
    - Execute the following command to test with the default model (walking.fbx):

```bash
docker run --gpus all --rm -i -u $(id -u):$(id -g) -v "$(pwd):/workspace" local/hydra_kubric python3 parseworker.py assets='fbx'
```

3. You can also use custom param `fbx_path='your/path/to/model.fbx'` to read your model

### Importing Custom Camera Path File

1. Create a folder named camera_path in the main directory if it doesn't already exist.
   Place your camera path file in this folder.

    - By default, a camera path file [path.txt](camera_path%2Fpath.txt) is provided in the folder for testing purposes.

2. Testing with the Default Path File
    - Execute the following command to test with the default path file (path.txt):

```bash
docker run --gpus all --rm -i -u $(id -u):$(id -g) -v "$(pwd):/workspace" local/hydra_kubric python3 parseworker.py camera='path'
```

3. To use a custom path file, specify the path to your file using the camera_path parameter:

```bash
camera='path' camera_path='your/path/to/path.txt'
```

## Docker Setup for Testing

This guide will assist you in setting up Docker for testing and evaluation purposes.

**Step-by-Step Guide:**

1. **Generate Docker File**
    - Navigate to the `dockers` sub-directory and run the following command to build the Docker image.
      Name the image `local/evaluate_flow`:

```bash
docker build -f Evaluation.Dockerfile -t local/evaluate_flow .
```

2. After the Docker image is successfully built, return to the main directory (where this README is located).

3. **Running the Test Code**
    - To run the test code, Docker needs access to three specific file locations:
        - The code location and the hydra configuration.
        - The TensorFlow Datasets (tfds) data location.
        - The location of the optical flow methods.
    - Additionally, ensure that Docker has access to the GPU.

## Generating a TensorFlow Dataset (TFDS)

**Prerequisites:**

- Created a docker image named `local/evaluate_flow` in previous steps
- Adjusted `scenes/flow_dataset.py` with your configs

**Step-by-Step Guide:**

1. Navigate to the main directory (where this README is located)
2. Run a docker container with following command:

```bash
docker run --gpus all --rm -it -u root -v "$(pwd):/workspace" local/evaluate_flow
```

3. Initialize Dataset Structure:
    - Execute `tfds new flow_dataset` in the docker container command line.
    - This creates a `flow_dataset/` sub-directory and needed files for the build process.

4. Copy the adjusted `scenes/flow_dataset.py` into the newly created directory,
   replacing the auto-generated file (flow_dataset/flow_dataset_dataset_builder.py).

5. Build the TFDS (This step may take some time):
    - Run `tfds build` within the new directory: `flow_dataset/`.
    - This generates TFRecords from the data and stores them in a sub-directory under `~/tensorflow_datasets`.
    - For Windows users, this will be in `C:\Users\your_name\`.

6. Perform a test by running `python test_tfds.py` in `/workspace`.

## Evaluation with Optical Flow Models

### ARFlow

1. Get the model from [ARFlow — Official PyTorch Implementation](https://github.com/lliuz/ARFlow) repository.
2. Extract the `ARFlow/` folder to the root directory of the flowdataset project.
3. Update the line `cxx_args = ['-std=c++11']` to `cxx_args = ['-std=c++17']`
   in `ARFlow/models/correlation_package/setup.py`
4. Start the evaluation Docker container using the usual command mentioned above.
5. In the active container, navigate to the correlation package directory and run the setup script:

```bash
cd ./ARFlow/models/correlation_package && python3 setup.py install
```

6. If the build fails or the correlation package is not found, modify the import lines in `models/pwclite.py` to ensure
   proper usage.
7. To run the evaluation, execute the following command in the `/workspace` directory:

```bash
python evaluate_flow.py
``` 

8. To save evaluation figures (EPE, flow, etc.), use:

```bash
python evaluate_flow.py figure='show'
```

