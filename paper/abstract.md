---
title: "Customizable Motion Scenario-based Dataset Generation for Optical Flow Evaluation"
authors:
  - Niloofar Hooshyaripour
  - Kangwei Liao
  - Jochen Lang
  - Emil M. Petriu
date: 2025
---

# Abstract

Optical flow estimation is a fundamental task in computer vision with diverse applications, including autonomous driving, video analysis, and robotics. Over the past decade, significant progress has been made through the development of deep learning models and synthetic datasets, enabling improved accuracy in estimating motion between frames. However, existing datasets often lack systematic motion scenario coverage, occlusion handling, and diverse environmental conditions, limiting their effectiveness for robust model evaluation. 

In this work, we introduce a novel video dataset—generated using the Kubric simulation environment—that addresses these gaps by offering **customizable motion scenarios** combining dynamic object interactions, camera movements, and environmental effects (e.g., lens distortion, raindrops, occlusions). Our dataset is organized by motion type and environmental challenge, enabling precise evaluation of optical flow models under controlled yet diverse conditions. 

We benchmark several state-of-the-art models (RAFT, FlowFormer, FlowNet, PWC-Net) on this dataset and demonstrate that transformer-based methods generalize better to unseen motion patterns and occlusions, while simpler CNN-based models struggle in complex scenes. Furthermore, fine-tuning on our dataset yields significant robustness improvements, confirming its value as a comprehensive benchmark for future optical flow research. :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
