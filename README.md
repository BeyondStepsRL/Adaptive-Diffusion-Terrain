# Adaptive Diffusion Terrain Generation

**[CoRL 24] Adaptive Diffusion Terrain Generation for Uneven Terrain Generation**
Youwei Yu$^\dagger$, Junhong Xu$^\dagger$, Lantao Liu
[[paper]](https://openreview.net/forum?id=xYleTh2QhS)[[arXiv]](https://arxiv.org/abs/2410.10766v1)[[project page]](https://www.youwei-yu.com/adtg-sim-to-real)


#### Environment Setup
```shell
git clone https://github.com/youwyu/Adaptive-Diffusion-Terrain.git
```
1. Isaac Gym, DDPM, Python3.8-dev (Make sure you have mini/ana conda installed)
```shell
. install.sh  ## Make sure using . rather than bash or sh install.sh
```

2. Semi-Global Matching on GPU
Change the CUDA path in contexts/simsense/setup.py Line#35
```shell
cd contexts/simsense
pip install .
```

#### Teacher & Student Policy
If you wanna use wandb, change Line#119, #120 in auto_train
```shell
python3 auto_train.py
```
Notes:
1. Terrain context will auto save as json file.
2. Teacher: specify the file to load the checkpoint, o.w. it will train from 0.
3. Student: it will auto find the json, or user specify json path. o.w. the program return 1.
4. We use a single RTX 4090 with 24GB RAM. For smaller RAM, we suggest lower num_agents_per_terrain and num_agents_per_terrain_distill in cfg/base_config.
   The number can be estimated roughly as YOUR_RAM * 4.
5. If you don't want privileged knowlege and save training time and RAM, set all use_globalmap to False.

#### Miscell
We understand the code is fully non-optimized as we do not care about simulation training during our bed time.
We kindly ask you to cite our work if you leverage the code.
```
@inproceedings{
   yu2024adaptive,
   title={\href{https://openreview.net/forum?id=xYleTh2QhS}{Adaptive Diffusion Terrain Generator for Autonomous Uneven Terrain Navigation}},
   author={Youwei Yu and Junhong Xu and Lantao Liu},
   booktitle={8th Annual Conference on Robot Learning},
   year={2024}
}
```