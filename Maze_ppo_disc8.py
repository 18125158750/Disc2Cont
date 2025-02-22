import argparse
import os
import random
import time
from datetime import datetime
import subprocess
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions.categorical import Categorical # 导入 Categorical 分布

def strtobool(s): # Replacement for strtobool
    if isinstance(s, bool):
        return s
    s = s.lower()
    return s in ('yes', 'true', 't', 'y', '1')

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # 实验配置
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="实验名称，用于命名 runs 目录和 WandB runs")
    parser.add_argument("--seed", type=int, default=1,
        help="随机种子，用于实验的可重复性")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="是否使用 torch 的确定性算法 (torch.backends.cudnn.deterministic=False)")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="是否使用 CUDA")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="是否使用 Weights and Biases (WandB) 跟踪实验")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-maze-discrete", # 修改 WandB 项目名称
        help="WandB 项目名称")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="WandB 实体 (团队)，用于组织项目")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="是否捕获 agent 表现的视频")
    parser.add_argument("--video-interval", type=int, default=100,
        help="视频捕获的频率，每 video-interval 个 updates 捕获一次")

    # 算法 (PPO) 超参数
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, # 修改默认学习率
        help="优化器的学习率")
    parser.add_argument("--total-timesteps", type=int, default=5e8,
        help="实验的总时间步数")
    parser.add_argument("--num-envs", type=int, default=1024,
        help="并行环境的数量")
    parser.add_argument("--num-steps", type=int, default=256,
        help="每个环境每次策略 rollout 运行的步数")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="是否对策略和价值网络的学习率进行退火")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="是否使用广义优势估计 (GAE) 计算优势函数")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="折扣因子 gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="GAE 的 lambda 参数")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="每个 epoch 的 mini-batches 数量")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="策略更新的 epochs 数量")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="是否标准化优势函数")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="PPO 裁剪系数")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="是否使用裁剪的价值函数损失")
    parser.add_argument("--ent-coef", type=float, default=0.05,
        help="熵正则化系数")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="价值函数损失系数")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="梯度裁剪的最大范数")
    parser.add_argument("--target-kl", type=float, default=None,
        help="目标 KL 散度阈值，用于早停")

    # 存储配置
    parser.add_argument("--save-freq", type=int, default=200,
        help="模型检查点保存频率 (updates)")
    parser.add_argument("--save-best", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="是否保存最佳模型")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs): #envs is now MazeEnv, assume it has discrete action space
        super(Agent, self).__init__()
        obs_shape = envs.observation_space.shape[0] # Get observation shape from MazeEnv
        n_actions = envs.action_space.n # Get number of discrete actions from MazeEnv

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 64)),
            nn.ELU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 64)),
            nn.ELU(),
            layer_init(nn.Linear(64, n_actions), std=0.01), # Output layer for discrete actions
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x) # Actor 输出 logits
        probs = Categorical(logits=logits) # 使用 Categorical 分布
        if action is None:
            action = probs.sample() # 从 Categorical 分布中采样动作
        return action, probs.log_prob(action), probs.entropy(), self.critic(x) # 返回动作，log 概率，熵和价值


import atexit
def exit_handler():
    print("Exiting program...")
    if args.track:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # 实验目录设置
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") # 生成时间戳，用于区分实验 runs
    run_name = f"{args.exp_name}-{timestamp}" # 实验 runs 名称，包含实验名和时间戳
    dirs = { # 定义实验相关的目录
        'base': f"runs/MazeDisc8/{run_name}", # 实验 runs 根目录
        'checkpoints': f"runs/MazeDisc8/{run_name}/checkpoints", # 检查点保存目录
        'best': f"runs/MazeDisc8/{run_name}/best_models", # 最佳模型保存目录
        'config': f"runs/MazeDisc8/{run_name}/config", # 配置文件保存目录
        'videos': f"runs/MazeDisc8/{run_name}/videos" # 视频录制保存目录
    }
    for d in dirs.values(): # 循环创建所有目录
        os.makedirs(d, exist_ok=True) # exist_ok=True 表示目录已存在时不会报错
    # 保存配置信息
    config = { # 存储实验配置信息
        'args': vars(args), # 命令行参数
        'git': { # Git 信息，方便追溯实验代码版本
            'commit': subprocess.getoutput('git rev-parse HEAD'), # 获取当前 commit ID
            'branch': subprocess.getoutput('git branch --show-current'), # 获取当前分支名
            'dirty': subprocess.getoutput('git diff --shortstat') # 检查是否有未提交的修改
        },
        'system': { # 系统信息
            'cuda_available': torch.cuda.is_available(), # CUDA 是否可用
            'device': str(device) # 使用的设备 (cuda 或 cpu)
        }
    }
    with open(f"{dirs['config']}/config.yaml", 'w') as f: # 将配置信息保存到 YAML 文件
        yaml.dump(config, f, indent=2) # indent=2 表示缩进为 2 个空格

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=False, # Removed monitor_gym as it's not standard gym env
            save_code=True,
        )
        # 注册退出处理程序
        atexit.register(exit_handler)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    # Import the Maze environment
    from env.Maze import Discrete4Maze,Discrete8Maze,Discrete16Maze
    step_length = 0.1
    envs = Discrete8Maze(num_envs=args.num_envs,
                        render_mode="rgb_array" if args.capture_video else None,
                        device=device,
                        step_length=step_length,
                        # start_pos=(8.5, 3.5),
                        start_pos=None,
                        exploration_reward = 0.0,
                        action_repeat_probability=0.10,
                        timeout=256,
                        success_rate_influence=0.0,
                        )
    assert isinstance(envs.action_space, gym.spaces.Discrete), "Only discrete action space is supported" # 确保环境是离散动作空间

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    total_episodes = 0
    episode_stats = {
        'returns': [],
        'lengths': [],
        'successes': 0,
        'collisions': 0
    }
    best_score = -np.inf
    last_msr = 0

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device) # Get obs shape from MazeEnv
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device) # 存储离散动作，类型为 long
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset() # MazeEnv reset returns obs and info
    next_obs = torch.Tensor(next_obs).to(device) # Ensure obs is tensor and on device
    next_done = torch.zeros(args.num_envs).to(device) # Initialize done states
    num_updates = int(args.total_timesteps // args.batch_size)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs # Increment global step by num_envs

            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action # 存储离散动作
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action) # MazeEnv step returns terminated and truncated
            rewards[step] = reward.to(device).view(-1)
            next_obs = next_obs.to(device) # Ensure next_obs is tensor and on device
            next_done = (torch.logical_or(terminated,truncated)).to(device) # Use terminated as done

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
                msr = infos['MASR']

        # 视频录制控制
        if args.capture_video and (update % args.video_interval == 0):
            envs.start_recording(video_path=dirs['videos']+f"/video_{update}.mp4")

        # 计算统计指标
        if total_episodes > 0:
            avg_return = np.mean(episode_stats['returns']) # 平均 episode 回报
            avg_length = np.mean(episode_stats['lengths']) # 平均 episode 长度
            max_length = np.max(episode_stats['lengths']) # 最大 episode 长度
            success_rate = episode_stats['successes'] / total_episodes # 成功率
            collision_rate = episode_stats['collisions'] / total_episodes # 碰撞率
        else:
            avg_return = avg_length = success_rate = collision_rate = 0

        # 计算训练速度
        sps = int(global_step / (time.time() - start_time)) # Samples Per Second

        # 格式化输出
        print(f"Update {update}/{num_updates+1} | " # 打印 update 轮数信息
            f"Episodes: {total_episodes} | " # 打印累计 episode 数量
            f"Avg Return: {avg_return:.2f} | " # 打印平均 episode 回报
            f"Avg Length: {avg_length:.2f} | " # 打印平均 episode 长度
            f"Max Length: {max_length:.2f} | " # 打印最大 episode 长度
            f"Success: {success_rate:.2%} | " # 打印成功率
            f"MSR: {msr:.2%} | " # 打印成功率
            f"Collision: {collision_rate:.2%} | " # 打印碰撞率
            f"SPS: {sps}") # 打印训练速度

        # WandB 记录训练数据
        if args.track:
            wandb.log({
                "charts/avg_return": avg_return,
                "charts/avg_episode_length": avg_length,
                "charts/max_episode_length": max_length,
                "charts/success_rate": success_rate,
                "charts/collision_rate": collision_rate,
                "charts/SPS": sps,
                "losses/learning_rate": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
                "update": update,
            }, step=global_step)
            if msr != last_msr:
                wandb.log({
                    "charts/MSR": msr,
                }, step=global_step)
                last_msr = msr

        # 重置统计
        episode_stats = {k: [] if isinstance(v, list) else 0 for k, v in episode_stats.items()}
        total_episodes = 0


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.to(torch.float)
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape) # Get obs shape from MazeEnv
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1) # 离散动作空间，action shape 变为 (batch_size,)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds]) # 传入离散动作
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.track:
            wandb.log({ # Log remaining metrics to wandb
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "charts/SPS": int(global_step / (time.time() - start_time))
            }, step=global_step)

        # 保存检查点
        if update % args.save_freq == 0: # 每隔 save_freq 轮 update 保存一次检查点
            checkpoint = { # 构建检查点字典
                'model': agent.state_dict(), # 模型参数
                'optimizer': optimizer.state_dict(), # 优化器状态
                'args': vars(args), # 训练参数
                'timestamp': timestamp # 实验时间戳
            }
            torch.save(checkpoint, f"{dirs['checkpoints']}/update_{update}.pt") # 保存检查点到文件
            # 保存最佳模型
            current_score = avg_return # 使用平均 episode 回报作为评估指标
            if args.save_best and current_score > best_score: # 如果 save_best 为 True 且当前分数超过最佳分数
                best_score = current_score # 更新最佳分数
                torch.save(checkpoint, f"{dirs['best']}/best_model.pt") # 保存最佳模型

    envs.close()
    if args.track:
        wandb.finish()