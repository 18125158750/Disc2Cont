import argparse
import os
import random
import time
import subprocess
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import yaml
import gymnasium as gym
torch.set_printoptions(precision=3, sci_mode=False)

# 自定义环境模块
def make_env(num_envs, device, render_mode=None, step_length=0.2):
    from env.Maze import ContinuousMaze
    return ContinuousMaze(
        num_envs=num_envs,
        device=device,
        start_pos=(8.5, 5.5),
        # start_pos=None,
        end_pos=(8.5, 8.5),
        step_length=step_length,
        render_mode=render_mode,
    )

@torch.no_grad()
def calculate_advantage(
    values: torch.Tensor,
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    steps_per_iteration: int,
    discount: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(steps_per_iteration)):
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1]

        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + discount * gae_lambda * nextnonterminal * lastgaelam
        )
    returns = advantages + values
    return advantages, returns

class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, 512)),
            nn.ELU(),
            self.layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            self.layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            self.layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, 512)),
            nn.ELU(),
            self.layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            self.layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            self.layer_init(nn.Linear(128, action_dim), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def parse_args():
    parser = argparse.ArgumentParser()
    # 实验配置
    parser.add_argument("--exp-name", type=str, default="PPO-ContinuousMaze")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(int(x)), default=True)
    parser.add_argument("--cuda", type=lambda x: bool(int(x)), default=True)
    parser.add_argument("--track", type=lambda x: bool(int(x)), default=False)
    parser.add_argument("--wandb-project", type=str, default="continuous-maze")
    
    # 训练参数
    parser.add_argument("--total-timesteps", type=int, default=2e9)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    
    # 存储配置
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    parser.add_argument("--save-best", type=lambda x: bool(int(x)), default=True)

    parser.add_argument("--capture-video", type=lambda x: bool(int(x)), default=True,)
    parser.add_argument("--video-interval", type=int, default=50,)
    
    args = parser.parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # 实验目录设置
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.exp_name}-{timestamp}"
    dirs = {
        'base': f"runs/{run_name}",
        'checkpoints': f"runs/{run_name}/checkpoints",
        'best': f"runs/{run_name}/best_models",
        'config': f"runs/{run_name}/config",
        'videos': f"runs/{run_name}/videos"
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 保存配置信息
    config = {
        'args': vars(args),
        'git': {
            'commit': subprocess.getoutput('git rev-parse HEAD'),
            'branch': subprocess.getoutput('git branch --show-current'),
            'dirty': subprocess.getoutput('git diff --shortstat')
        },
        'system': {
            'cuda_available': torch.cuda.is_available(),
            'device': str(device)
        }
    }
    with open(f"{dirs['config']}/config.yaml", 'w') as f:
        yaml.dump(config, f, indent=2)

    # 初始化环境
    step_length = 0.2
    env = make_env(
        args.num_envs, 
        device,
        # render_mode='human',
        step_length=step_length
    )
    agent = Agent(obs_dim=2, action_dim=2).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # 训练状态初始化
    best_score = -np.inf
    global_step = 0
    start_time = time.time()

    total_episodes = 0
    episode_stats = {
        'returns': [],
        'lengths': [],
        'successes': 0,
        'collisions': 0
    }
    
    # 存储缓冲区
    obs = torch.zeros((args.num_steps, args.num_envs, 2), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, 2), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    
    # 训练主循环
    next_obs, _ = env.reset()
    next_done = torch.zeros(args.num_envs, device=device)
    
    updates = int(args.total_timesteps // args.batch_size) + 1
    for update in range(1, updates):
        # 数据收集阶段
        for step in range(args.num_steps):
            global_step += args.num_envs
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob
            obs[step] = next_obs
            
            # 环境交互
            next_obs, reward, done, _, infos = env.step(action)
            rewards[step] = reward
            dones[step] = next_done
            next_done = done.to(device)


            # 处理完成环境的信息
            if 'final_info' in infos:
                for info in infos['final_info']:
                    total_episodes += 1
                    episode_stats['lengths'].append(info['steps'])
                    episode_stats['returns'].append(info['returns'])
                    if info['success']:
                        episode_stats['successes'] += 1
                    if info['collision']:
                        episode_stats['collisions'] += 1
            
        # 视频录制控制
        if args.capture_video and (update % args.video_interval == 0):
            env.start_recording(video_path=dirs['videos']+f"/video_{update}.mp4")
        
        # 计算统计指标
        if total_episodes > 0:
            avg_return = np.mean(episode_stats['returns'])
            avg_length = np.mean(episode_stats['lengths'])
            max_length = np.max(episode_stats['lengths'])
            success_rate = episode_stats['successes'] / total_episodes
            collision_rate = episode_stats['collisions'] / total_episodes
        else:
            avg_return = avg_length = success_rate = collision_rate = 0
        
        # 计算训练速度
        sps = int(global_step / (time.time() - start_time))
        
        # 格式化输出
        print(f"Update {update}/{updates} | "
            f"Episodes: {total_episodes} | "
            f"Avg Return: {avg_return:.2f} | "
            f"Avg Length: {avg_length:.2f} | "
            f"Max Length: {max_length:.2f} | "
            f"Success: {success_rate:.2%} | "
            f"Collision: {collision_rate:.2%} | "
            f"SPS: {sps}")
        
        # 重置统计
        episode_stats = {k: [] if isinstance(v, list) else 0 for k, v in episode_stats.items()}
        total_episodes = 0
        
        # 计算优势函数和回报
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages, returns = calculate_advantage(
            values,
            next_value,
            rewards,
            dones,
            next_done,
            args.num_steps,
            args.gamma,
            args.gae_lambda,
        )

        # 策略优化
        b_obs = obs.reshape(-1, obs.shape[-1])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, actions.shape[-1])
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        
        early_stop = False
        for epoch in range(args.update_epochs):
            perm = torch.randperm(args.batch_size, device=device)
            for i in range(0, args.batch_size, args.minibatch_size):
                idx = perm[i:i+args.minibatch_size]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[idx], b_actions[idx])
                logratio = newlogprob - b_logprobs[idx]
                ratio = logratio.exp()
                
                # 策略损失
                pg_loss = -b_advantages[idx] * ratio
                pg_loss = torch.clamp(pg_loss, 1-args.clip_coef, 1+args.clip_coef).mean()
                
                # 价值损失
                v_loss = 0.5 * (newvalue.flatten() - b_returns[idx]).pow(2).mean()
                
                # 总损失
                loss = pg_loss + v_loss * 0.5 - entropy.mean() * args.ent_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    if approx_kl > 0.02:  # 早停机制
                        print(
                            f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.4f} > 0.02"
                        )
                        early_stop = True
            if early_stop:
                break

        # 保存检查点
        if update % args.save_freq == 0:
            checkpoint = {
                'model': agent.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'timestamp': timestamp
            }
            torch.save(checkpoint, f"{dirs['checkpoints']}/update_{update}.pt")
            
            # 保存最佳模型
            current_score = avg_return
            if args.save_best and current_score > best_score:
                best_score = current_score
                torch.save(checkpoint, f"{dirs['best']}/best_model.pt")
                
        # # 打印训练信息
        # print(f"Update {update}/{args.total_timesteps//args.batch_size} | "
        #      f"Return: {returns.mean().item():.2f} | "
        #      f"Time: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    main()