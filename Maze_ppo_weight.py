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
import torch.nn.functional as F # Import functional
from torch.distributions.normal import Normal

def strtobool(s): # Replacement for strtobool
    if isinstance(s, bool):
        return s
    s = s.lower()
    return s in ('yes', 'true', 't', 'y', '1')
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Experiment Config
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="experiment name, used in runs folder and wandb run name")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-maze-dirichlet",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check `videos` folder)")
    parser.add_argument("--video-interval", type=int, default=100,
        help="video capture interval (updates)")

    # Algorithm (PPO) Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-5,
        help="optimizer learning rate")
    parser.add_argument("--total-timesteps", type=int, default=5e8,
        help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=1024,
        help="number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
        help="number of steps to rollout in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="Discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="Lambda for GAE")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="Number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="K epochs for update")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantage normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="Ratio clip coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles clipped value function loss")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="Maximum norm of the gradients")
    parser.add_argument("--target-kl", type=float, default=None,
        help="Target KL divergence threshold")

    # Dirichlet Policy Specific Arguments
    parser.add_argument("--num-base-outputs", type=int, default=4,
        help="Number of base actions to weight (e.g., 4 for up/down/left/right)")


    # Storage arguments
    parser.add_argument("--save-freq", type=int, default=200,
        help="checkpoint save frequency (updates)")
    parser.add_argument("--save-best", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save best model")


    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SimpleDirichletPolicy(nn.Module): # Renamed and simplified policy class
    def __init__(self, envs, device,
                 num_base_outputs=4):
        super().__init__()
        self.device = device
        obs_dim = envs.observation_space.shape[0]
        self.num_base_outputs = num_base_outputs

        # Network to directly output alpha logits
        self.alpha_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, num_base_outputs)), # Output num_base_outputs logits
        )

        # Value function network - input is just observation
        value_input_dim = obs_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

        # Temperature parameter (optional, can be removed if not needed)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.temperature_upper = 5.0
        self.temperature_lower = 0.1


    def to(self, device):
        self.alpha_net.to(device)
        self.value_head.to(device)
        return super().to(device)

    def get_value(self, obs):
        return self.value_head(obs)


    def sample_weights(self, obs, base_actions_list, eval_mode=False, weights=None): # base_actions_list is not used here anymore, but kept for interface compatibility
        alpha_logits = self.alpha_net(obs) # Directly get alpha logits
        alpha_logits = alpha_logits - torch.mean(alpha_logits, dim=-1, keepdim=True).detach()
        alpha = torch.exp(alpha_logits)
        # alpha = F.softplus(alpha_logits, beta=0.1) + 1e-6 # 加上一个小的正数防止 alpha 为 0

        # Temperature scaling (optional)
        # clamped_temp = self.temperature.clamp(
        #     min=self.temperature_lower,
        #     max=self.temperature_upper
        # )
        scaled_alpha = alpha / self.temperature
        # scaled_alpha = alpha*100


        weights_dist = torch.distributions.Dirichlet(scaled_alpha)

        if eval_mode:
            sampled_weights = alpha / alpha.sum(dim=-1, keepdim=True) # Deterministic weights: mean
        else:
            if weights is None:
                sampled_weights = weights_dist.sample()
            else:
                sampled_weights = weights
        log_prob = weights_dist.log_prob(sampled_weights) # Calculate log probability of sampled weights
        weights_dist_entropy = weights_dist.entropy()
        return sampled_weights, log_prob, weights_dist_entropy, alpha


    def get_naction_weighted(self, base_nactions_list, weights):
        weighted_action_mean = torch.zeros_like(base_nactions_list[0].repeat(weights.shape[0], 1)) # Initialize with shape of first base action
        for i in range(self.num_base_outputs):
            weighted_action_mean += weights[:, i:i+1] * base_nactions_list[i].repeat(weights.shape[0], 1)
        return weighted_action_mean

    @property
    def actor_parameters(self):
        actor_params = list(self.alpha_net.parameters())
        actor_params.append(self.temperature) # Keep temperature if used
        return actor_params

    @property
    def critic_parameters(self):
        return list(self.value_head.parameters())


import atexit
def exit_handler():
    print("Exiting program...")
    if args.track:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # Experiment directory setup
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.exp_name}-{timestamp}"
    dirs = {
        'base': f"runs/MazeDirichlet/{run_name}",
        'checkpoints': f"runs/MazeDirichlet/{run_name}/checkpoints",
        'best': f"runs/MazeDirichlet/{run_name}/best_models",
        'config': f"runs/MazeDirichlet/{run_name}/config",
        'videos': f"runs/MazeDirichlet/{run_name}/videos"
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    # Save configuration
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

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
        atexit.register(exit_handler)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Env setup
    from env.Maze import ContinuousMaze
    step_length = 0.2
    envs = ContinuousMaze(num_envs=args.num_envs,
                        render_mode=None, # Changed render_mode
                        device=device,
                        step_length=step_length,
                        start_pos=None,
                        exploration_reward = 0.0,
                        action_repeat_probability=0.0,
                        timeout=256,
                        )

    agent = SimpleDirichletPolicy(envs, device, # Using simplified policy
                                            num_base_outputs=args.num_base_outputs).to(device)
    optimizer_actor = optim.Adam(agent.actor_parameters, lr=args.learning_rate, eps=1e-5) # Separate optimizer for actor
    optimizer_critic = optim.Adam(agent.critic_parameters, lr=args.learning_rate, eps=1e-5) # Separate optimizer for critic


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
    obs_tensor = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device) # Rename obs to obs_tensor
    weights_tensor = torch.zeros((args.num_steps, args.num_envs, args.num_base_outputs)).to(device) # Store weights
    logprobs_tensor = torch.zeros((args.num_steps, args.num_envs)).to(device) # Rename logprobs to logprobs_tensor
    rewards_tensor = torch.zeros((args.num_steps, args.num_envs)).to(device) # Rename rewards to rewards_tensor
    dones_tensor = torch.zeros((args.num_steps, args.num_envs)).to(device) # Rename dones to dones_tensor
    values_tensor = torch.zeros((args.num_steps, args.num_envs)).to(device) # Rename values to values_tensor
    base_actions_tensor = torch.zeros((args.num_steps, args.num_envs, envs.action_space.shape[0])).to(device) # Store weighted base actions

    # Base actions definition (Up, Down, Left, Right)
    base_actions_list = [
        torch.tensor([0, step_length], dtype=torch.float32).to(device), # Up
        torch.tensor([0, -step_length], dtype=torch.float32).to(device), # Down
        torch.tensor([-step_length, 0], dtype=torch.float32).to(device), # Left
        torch.tensor([step_length, 0], dtype=torch.float32).to(device), # Right
    ]


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = int(args.total_timesteps // args.batch_size)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer_actor.param_groups[0]["lr"] = lrnow
            optimizer_critic.param_groups[0]["lr"] = lrnow


        for step in range(0, args.num_steps):
            global_step += args.num_envs

            obs_tensor[step] = next_obs
            dones_tensor[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                value = agent.get_value(next_obs) # Get value from agent
                values_tensor[step] = value.flatten() # Store value

                sampled_weights, logprob, _, _ = agent.sample_weights(next_obs, base_actions_list) # Sample weights using Dirichlet policy (base_actions_list is not used)
                weights_tensor[step] = sampled_weights # Store weights
                logprobs_tensor[step] = logprob # Store log probability

                naction = agent.get_naction_weighted(base_actions_list, sampled_weights) # Get weighted action
                base_actions_tensor[step] = naction # Store weighted action


            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(naction.cpu().numpy()) # Step with weighted action
            rewards_tensor[step] = reward.to(device).view(-1)
            next_obs = next_obs.to(device)
            next_done = (torch.logical_or(terminated, truncated)).to(device)

            # Handle episode termination info
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

        # Video recording control
        if args.capture_video and (update % args.video_interval == 0):
            envs.start_recording(video_path=dirs['videos']+f"/video_{update}.mp4")

        # Calculate statistics
        if total_episodes > 0:
            avg_return = np.mean(episode_stats['returns'])
            avg_length = np.mean(episode_stats['lengths'])
            max_length = np.max(episode_stats['lengths'])
            success_rate = episode_stats['successes'] / total_episodes
            collision_rate = episode_stats['collisions'] / total_episodes
        else:
            avg_return = avg_length = success_rate = max_length = collision_rate = 0

        sps = int(global_step / (time.time() - start_time))

        print(f"Update {update}/{num_updates+1} | "
            f"Episodes: {total_episodes} | "
            f"Avg Return: {avg_return:.2f} | "
            f"Avg Length: {avg_length:.2f} | "
            f"Max Length: {max_length:.2f} | "
            f"Success: {success_rate:.2%} | "
            f"MSR: {msr:.2%} | "
            f"Collision: {collision_rate:.2%} | "
            f"SPS: {sps}")

        if args.track:
            wandb.log({
                "charts/avg_return": avg_return,
                "charts/avg_episode_length": avg_length,
                "charts/max_episode_length": max_length,
                "charts/success_rate": success_rate,
                "charts/collision_rate": collision_rate,
                "charts/SPS": sps,
                "losses/learning_rate": optimizer_actor.param_groups[0]["lr"],
                "global_step": global_step,
                "update": update,
            }, step=global_step)
            if msr != last_msr:
                wandb.log({
                    "charts/MSR": msr,
                }, step=global_step)
                last_msr = msr


        episode_stats = {k: [] if isinstance(v, list) else 0 for k, v in episode_stats.items()}
        total_episodes = 0

        # Value bootstrapping
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards_tensor).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.to(torch.float)
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_tensor[t + 1].to(torch.float)
                        nextvalues = values_tensor[t + 1]
                    delta = rewards_tensor[t] + args.gamma * nextvalues * nextnonterminal - values_tensor[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values_tensor
            else:
                returns = torch.zeros_like(rewards_tensor).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones_tensor[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards_tensor[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values_tensor

        # Flatten the batch
        b_obs = obs_tensor.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs_tensor.reshape(-1)
        b_weights = weights_tensor.reshape((-1, args.num_base_outputs)) # Flatten weights
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_tensor.reshape(-1)


        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                value = agent.get_value(b_obs[mb_inds]).view(-1) # Get value from agent
                _, newlogprob, entropy, _ = agent.sample_weights(b_obs[mb_inds], base_actions_list, weights=b_weights[mb_inds]) # Pass weights for logprob calculation

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
                if args.clip_vloss:
                    v_loss_unclipped = (value - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        value - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer_actor.zero_grad() # Zero actor optimizer gradients
                optimizer_critic.zero_grad() # Zero critic optimizer gradients
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer_actor.step() # Step actor optimizer
                optimizer_critic.step() # Step critic optimizer


            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.track:
            wandb.log({
                "charts/learning_rate_actor": optimizer_actor.param_groups[0]["lr"], # Log actor lr
                "charts/learning_rate_critic": optimizer_critic.param_groups[0]["lr"], # Log critic lr
                "losses/total_loss": loss.item(),
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "histograms/sampled_weights": wandb.Histogram(weights_tensor.cpu()), # Log sampled weights
                "values/value": value.mean().item(), # Log value
                "values/advantages": mb_advantages.mean().item(), # Log advantages
                "values/returns": b_returns.mean().item(), # Log returns
                "charts/temperature": agent.temperature.item(),

            }, step=global_step)

        if update % args.save_freq == 0:
            checkpoint = {
                'model_state_dict': agent.state_dict(), # Save agent state dict
                'optimizer_actor_state_dict': optimizer_actor.state_dict(), # Save actor optimizer
                'optimizer_critic_state_dict': optimizer_critic.state_dict(), # Save critic optimizer
                'args': vars(args),
                'timestamp': timestamp,
            }
            torch.save(checkpoint, f"{dirs['checkpoints']}/update_{update}.pt")
            current_score = avg_return
            if args.save_best and current_score > best_score:
                best_score = current_score
                torch.save(checkpoint, f"{dirs['best']}/best_model.pt")

    envs.close()
    if args.track:
        wandb.finish()