# Adaptive Diffusion Terrain Generation

**[RSS 25 ROAR] ADEPT: Adaptive Diffusion Environment for Policy Transfer Sim-to-Real**

[[paper]](https://openreview.net/forum?id=tdgQT1SynU) [[arXiv]](https://arxiv.org/abs/2506.01759)

**[CoRL 24] Adaptive Diffusion Terrain Generation for Autonomous Uneven Terrain Navigation**

[[paper]](https://openreview.net/forum?id=xYleTh2QhS) [[arXiv]](https://arxiv.org/abs/2410.10766v1)

*[Youwei Yu\*](https://www.youwei-yu.com), [Junhong Xu\*](https://junhongxu.github.io), [Lantao Liu](https://vail.sice.indiana.edu/pages/lantaoliu.html)*

[[project page]](https://www.youwei-yu.com/adtg-sim-to-real)

### TODO
- [ ] Add consistency diffusion-based 3D wild environment generation
- [ ] Add standalone code of environment generation for Isaac Gym, Isaac Lab, Mujoco Playground, and Gazebo

### Environment Setup
```shell
git clone https://github.com/youwyu/Adaptive-Diffusion-Terrain.git
```
1. Isaac Gym, DDPM, Python3.8-dev (Make sure you have mini/ana-conda installed)
```shell
. install.sh  ## Make sure using . rather than bash or sh install.sh
```

2. Semi-Global Matching on GPU

Make sure the CMake version is at least 3.18, otherwise install by Kitware at https://apt.kitware.com or build from source
```shell
wget https://github.com/Kitware/CMake/releases/download/v3.31.0/cmake-3.31.0.tar.gz
tar -xvf cmake-3.31.0.tar.gz
cd cmake-3.31.0
./configure
make
sudo make install
```

Change the CUDA path in contexts/simsense/setup.py Line#35
```shell
pip install contexts/simsense
```

### Teacher & Student Policy
If you wanna use wandb, change Line#119, #120 in auto_train
```shell
python3 auto_train.py
```
Notes:
- Terrain context will auto-save as json file.
- Teacher: specify the file to load the checkpoint, o.w. it will train from 0.
- Student: it will auto-find the json, or the user will specify json path. o.w. the program returns 1.
- We use a single RTX 4090 with 24GB RAM. For smaller RAM, we suggest lower num_agents_per_terrain and num_agents_per_terrain_distill in cfg/base_config.
   The number can be estimated roughly as YOUR_RAM * 4.
- If you don't want privileged knowledge and save training time and RAM, set all use_globalmap to False.

### Miscell
Please consider cite our work if it helps your sim-to-real training.
```
@inproceedings{
yu2025adept,
   title={\href{https://openreview.net/forum?id=tdgQT1SynU}{{ADEPT}: Adaptive Diffusion Environment for Policy Transfer Sim-to-Real}},
   author={Youwei Yu and Junhong Xu and Lantao Liu},
   booktitle={RSS 2025 Workshop on Resilient Off-road Autonomous Robotics},
   year={2025}
}

@inproceedings{
   yu2024adaptive,
   title={\href{https://openreview.net/forum?id=xYleTh2QhS}{Adaptive Diffusion Terrain Generator for Autonomous Uneven Terrain Navigation}},
   author={Youwei Yu and Junhong Xu and Lantao Liu},
   booktitle={8th Annual Conference on Robot Learning},
   year={2024}
}
```
