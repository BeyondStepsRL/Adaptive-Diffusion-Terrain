from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 16,
    channels = 1,
    # flash_attn = True,
    dim_mults = (1, 2, 4)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    # loss_type = 'l1'            # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    '../../assets/elevation/train/',
    train_batch_size = 128,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    save_and_sample_every=500,
    results_folder='results',
)
# trainer.load(45)
trainer.train()