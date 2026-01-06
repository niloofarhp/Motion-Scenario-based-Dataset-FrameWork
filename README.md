# Motion-Scenario-based-Dataset-FrameWork
This Repository presents a customizable Motion Scenario-based Dataset Generation FrameWork. 

This FrameWork is a Kubric-based toolkit for generating customizable video sequences with ground-truth optical flow, designed to benchmark and train modern optical flow models. Unlike existing datasets, it systematically covers diverse motion scenarios—combining object trajectories, camera egomotion, environmental effects (rain, fisheye distortion), and occlusion via “bar blockers”.

## Key Features  
- **Scenario-Based Organization**  
  - Object motions: static, sliding, rotating  
  - Camera motions: fixed, linear, Bézier-path, circular 
- **Environmental Effects**  
  - Fisheye lens distortion  
  - Procedural rain (texture- or shader-based)  
- **Occlusion Simulation**  
  - Texture-based bar blockers to introduce depth/motion discontinuities  
- **Fully Configurable via Hydra**  
  - Hierarchical configs in `conf/`  
  - Multirun support for parameter sweeps  

## Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/Motion-Scenario-based-Dataset-FrameWork.git
   cd code
2. Build a Docker Image
   ```bash
    cd code
    ./generateDocker.sh
3. Run a simple generte test
   ```bash
   docker run --gpus all --rm -u $(id -u):$(id -g) -v "$(pwd):/workspace" local/hydra_kubric python3 parseworker.py   
For more details about the framework implementation and generating your custom dataset refer to the [code folder](https://github.com/niloofarhp/Motion-Scenario-based-Dataset-FrameWork/tree/main/code).
 
## Available Dataset
  The Dataset we have generated is avialable using this [Link](https://uottawa-my.sharepoint.com/personal/nhoos082_uottawa_ca/_layouts/15/guestaccess.aspx?share=Em0aGJ-ycohDhyB9uH2gRvMB0WqmagwDsTIxuo8jfejl1g&e=EuYdMC)

