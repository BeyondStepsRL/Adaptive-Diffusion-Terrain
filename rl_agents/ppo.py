import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam

class PPO:
    def __init__(self, buffer, actor_critic_net, lr, ppo_epoch, mini_batch_size, 
                 clip_param, value_loss_coeff, entropy_coeff, tensorboard_writer, 
                 clip_grad_norm=1.0, min_std_value=0.1):
        self.buffer = buffer
        self.actor_critic_net = actor_critic_net
        self.lr = lr 
        self.ppo_epoch = ppo_epoch 
        # mini_batch_size is the number of samples in each training batch
        self.mini_batch_size = mini_batch_size
        self.clip_param = clip_param
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.optimizer = Adam(self.actor_critic_net.parameters(), lr=self.lr)
        self.tensorboard_writer = tensorboard_writer
        self.clip_grad_norm = clip_grad_norm
        self.use_clipped_value_loss = True
        self.min_std_value = min_std_value

    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic_net.compute_state_value(last_critic_obs).detach()
        self.buffer.compute_return_and_advantage(last_values)

    def update(self):
        batch_generator = self.buffer.sample_batch(self.ppo_epoch, self.mini_batch_size)
        pg_loss_epoch = [] 
        value_loss_epoch = []
        total_loss_epoch = [] 
        for state, action, old_mu_batch, old_sigma_batch, \
            value_targets, old_action_logprobs, advantage_fn in batch_generator:
            # compute the log probabilities of the action based on the current policy 
            new_action, new_action_logprob, new_state_values = self.actor_critic_net.evaluate(state, action)
            # compute the entropy of the policy 
            entropy = self.actor_critic_net.compute_entropy(state)
            mu_batch = self.actor_critic_net.action_dist.mean
            sigma_batch = torch.exp(self.actor_critic_net.logstd)

            # adjust the learning rate
            # if PPO_Args.desired_kl != None and PPO_Args.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                            torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                            2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > 0.01 * 2.0:
                    self.lr = max(1e-5, self.lr / 1.5)
                elif kl_mean < 0.01 / 2.0 and kl_mean > 0.0:
                    self.lr = min(1e-2, self.lr * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

            # 1. compute the surrogate clipped loss
            ratio = (new_action_logprob - old_action_logprobs).exp()
            surrogate_loss_1 = ratio * advantage_fn
            surrogate_loss_2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage_fn
            surrogate_loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean()

            # 2. compute the value loss
            if self.use_clipped_value_loss:
                value_clipped = value_targets + \
                                (new_state_values - value_targets).clamp(-self.clip_param,
                                                                          self.clip_param)
                value_losses = (new_state_values - value_targets).pow(2)
                value_losses_clipped = (value_clipped - value_targets).pow(2)
                value_fn_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_fn_loss = (value_targets - new_state_values).pow(2).mean()

            # total loss
            loss = surrogate_loss + self.value_loss_coeff * value_fn_loss - self.entropy_coeff * entropy.mean() # + self.actor_critic_net.compute_encoder_weight_decay()

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic_net.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            # clip the std of the actor critic
            self.actor_critic_net.logstd.data.clamp_(min=np.log(self.min_std_value))
            
            # record the losses
            pg_loss_epoch.append(surrogate_loss.item())
            value_loss_epoch.append(value_fn_loss.item())
            total_loss_epoch.append(loss.item())
        return np.mean(pg_loss_epoch), np.mean(value_loss_epoch), np.mean(total_loss_epoch), self.lr