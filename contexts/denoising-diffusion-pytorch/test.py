from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

model = Unet(
    dim = 16,
    channels = 1,
    dim_mults = (1, 2, 4)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 500,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    # loss_type = 'l1'            # L1 or L2
).cuda()
state_dict = torch.load('./results/model-3.pt')
diffusion.load_state_dict(state_dict['model'])
samples = diffusion.sample(30)

x = np.arange(0, 128)
y = np.arange(0, 128)
xx, yy = np.meshgrid(x, y)

for i in range(samples.size(0)):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    z = samples[i, 0, :, :].cpu().numpy()
    ax.plot_surface(xx, yy, z, cmap='coolwarm')
    fig.savefig('./' + str(i) + '.png')
    # np.savetxt('assets/test_map{}.txt'.format(i), z, delimiter=',')
    # plt.show()