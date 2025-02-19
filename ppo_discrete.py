import argparse
import os
import random
import time
from datetime import datetime
import gymnasium as gym
import gym_pusht
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter # Remove tensorboard import
import yaml # Import yaml for saving config

def str_to_bool(s): # Replacement for strtobool
    if isinstance(s, bool):
        return s
    s = s.lower()
    return s in ('yes', 'true', 't', 'y', '1')

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="gym_pusht/PushT-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=100000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=str_to_bool, default=True, nargs="?", const=True, # Use str_to_bool
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=str_to_bool, default=True, nargs="?", const=True, # Use str_to_bool
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=str_to_bool, default=False, nargs="?", const=True, # Use str_to_bool
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="pushT-ppo",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=str_to_bool, default=False, nargs="?", const=True, # Use str_to_bool
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=20,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=300,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=str_to_bool, default=True, nargs="?", const=True, # Use str_to_bool
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=str_to_bool, default=True, nargs="?", const=True, # Use str_to_bool
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.98,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=str_to_bool, default=True, nargs="?", const=True, # Use str_to_bool
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=str_to_bool, default=True, nargs="?", const=True, # Use str_to_bool
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-freq", type=int, default=500, # Add save frequency argument
        help="save frequency of model (number of updates)")
    parser.add_argument("--num-actions", type=int, default=4, choices=[4, 8],
                        help="Number of discrete actions: 4 or 8") # Add num_actions argument
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"runs/{run_name}/videos")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, num_actions):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_actions), std=0.01), # Output layer for discrete actions
        )
        self.num_actions = num_actions
        if num_actions == 4:
            self.action_space = torch.tensor([[20, 0], [0, 20], [-20, 0], [0, -20]], dtype=torch.float32)
        elif num_actions == 8:
            self.action_space = torch.tensor([[20, 0], [0, 20], [-20, 0], [0, -20], [20, 20], [20, -20], [-20, 20], [-20, -20]], dtype=torch.float32)
        else:
            raise ValueError("num_actions must be 4 or 8")
        self.obs_max = torch.tensor(envs.single_observation_space.high, dtype=torch.float32)
        self.obs_min = torch.tensor(envs.single_observation_space.low, dtype=torch.float32)

    def to(self, device):
        super().to(device)
        self.action_space = self.action_space.to(device)
        self.obs_max = self.obs_max.to(device)
        self.obs_min = self.obs_min.to(device)
        return self

    def get_value(self, x):
        return self.critic(self.normalize_obs(x))

    def get_action_and_value(self, x, action=None):
        logits = self.actor(self.normalize_obs(x))
        probs = Categorical(logits=logits)
        if action is None:
            action_index = probs.sample()
        else:
            action_index = action # Assume action is already the index
        continuous_action = self.action_space[action_index] # Convert discrete index to continuous action, directly index tensor
        return action_index, probs.log_prob(action_index), probs.entropy(), self.critic(x), continuous_action

    def normalize_obs(self, x):
        return (x - self.obs_min)*2 / (self.obs_max - self.obs_min) - 1


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Create working directory
    working_dir = f"runs/{run_name}"
    os.makedirs(working_dir, exist_ok=True)
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save arguments to yaml file
    config_path = os.path.join(working_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, indent=2)
    print(f"Configuration saved to: {config_path}")


    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False, # No tensorboard sync
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    print(envs.single_observation_space)
    print(envs.single_action_space)

    agent = Agent(envs, args.num_actions).to(device) # Pass num_actions to Agent
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) ).to(device) # Store discrete action indices
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Manual episode tracking
    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)
    ep_initial_reward = np.zeros(args.num_envs, dtype=np.float32) # Initialize ep_initial_reward
    ep_block_push_rewards = np.zeros(args.num_envs, dtype=np.float32) # Initialize ep_block_push_rewards
    is_first_step_in_ep = np.ones(args.num_envs, dtype=bool) # Flag to check first step
    prev_distance_to_goal = np.full(args.num_envs, float('inf')) # Initialize previous distance to goal with infinity

    # Store episode stats for each update
    update_ep_returns = []
    update_ep_lengths = []
    update_ep_successes = []
    update_ep_delta_rewards = [] # Initialize update_ep_delta_rewards
    update_ep_block_push_rewards = [] # Initialize update_ep_block_push_rewards


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    observation, info = envs.reset()
    prev_block_states = observation[:, 2:5].copy() # Initialize previous block states
    next_obs = torch.Tensor(observation).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    prev_block_states = observation[:, 2:5].copy() # Initialize previous block states after reset
    block_pose_init = observation[:, 2:5]
    goal_pose_init = np.tile(np.array([512, 512, 0.785398]), (args.num_envs, 1))
    prev_distance_to_goal[:] = np.sqrt(np.sum((block_pose_init[:, :2] - goal_pose_init[:, :2])**2, axis=1)) # Initialize prev_distance_to_goal in batch


    for update in range(1, num_updates + 1):
        # Clear episode stats for the current update
        update_ep_returns.clear()
        update_ep_lengths.clear()
        update_ep_successes.clear()
        update_ep_delta_rewards.clear() # Clear delta rewards list
        update_ep_block_push_rewards.clear() # Clear block push rewards list


        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_index, logprob, _, value, continuous_action = agent.get_action_and_value(next_obs) # Get discrete index and continuous action
                values[step] = value.flatten()
            actions[step] = action_index
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(torch.clamp(next_obs[:,:2]+continuous_action,min=0,max=512).cpu().numpy()) # Pass continuous action to env
            current_block_states = next_obs[:, 2:5].copy() # Get current block states
            goal_pose = next_obs[:, 5:8] # Get goal pose

            for env_id in range(args.num_envs): # Iterate through each environment
                if is_first_step_in_ep[env_id]:
                    ep_initial_reward[env_id] = reward[env_id] # Record reward at the beginning of the step
                    is_first_step_in_ep[env_id] = False # Set flag to False for subsequent steps

            # --- Batched Reward Calculation ---
            block_state_changed = ~np.all(current_block_states == prev_block_states, axis=1)
            block_state_changed = np.logical_or(block_state_changed, reward>0.95)
            block_push_reward = np.where(block_state_changed, 0.001, 0.0)

            goal_block_states = np.array([512, 512, 0.785398]) # Define goal block state
            distance = np.sqrt(np.sum((current_block_states[:, :2] - goal_block_states[:2])**2, axis=1))
            delta_angle = np.abs(current_block_states[:, 2] - goal_block_states[2])

            block_push_reward += np.where(block_state_changed, 0.1 * np.exp(-distance/100), 0.0)
            block_push_reward += np.where(block_state_changed, 0.1 * np.exp(-delta_angle/0.785398), 0.0)


            reward = reward * 0.0 # Discard environment reward

            reward += block_push_reward # Add block push reward to original reward

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            done = np.logical_or(terminated, truncated)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            prev_block_states = current_block_states.copy() # Update previous block states for next step


            # --- Batched Episode Tracking and Logging ---
            ep_returns += reward
            ep_lengths += 1
            ep_block_push_rewards += block_push_reward


            just_finished_episodes = done.nonzero()[0]
            for env_id in just_finished_episodes:
                delta_reward = ep_returns[env_id] - ep_initial_reward[env_id] * ep_lengths[env_id]
                update_ep_delta_rewards.append(delta_reward)
                update_ep_returns.append(ep_returns[env_id])
                update_ep_lengths.append(ep_lengths[env_id])
                update_ep_successes.append(info['is_success'][env_id])
                update_ep_block_push_rewards.append(ep_block_push_rewards[env_id])

                ep_returns[env_id] = 0
                ep_lengths[env_id] = 0
                ep_block_push_rewards[env_id] = 0
                is_first_step_in_ep[env_id] = True
                ep_initial_reward[env_id] = 0


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
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
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1) # Discrete actions are now just indices
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

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds]) # Get action and value, pass discrete action index
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

        # --- Update-level logging and printing ---
        avg_ep_return = np.mean(update_ep_returns) if update_ep_returns else np.nan
        avg_ep_length = np.mean(update_ep_lengths) if update_ep_lengths else np.nan
        success_rate = np.mean(update_ep_successes) if update_ep_successes else np.nan
        avg_adr = np.mean(update_ep_delta_rewards) if update_ep_delta_rewards else np.nan # Calculate avg ADR
        avg_ep_block_push_reward = np.mean(update_ep_block_push_rewards) if update_ep_block_push_rewards else np.nan # Calculate avg ep block push reward

        print(f"Update {update} Summary: "
            f"GS: {global_step:<8} "
            f"AR: {avg_ep_return:<8.2f} "
            f"AL: {avg_ep_length:<8.2f} "
            f"SR: {success_rate:<8.2f} "
            f"ADR: {avg_adr:<8.2f} " # Print ADR
            f"BPR: {avg_ep_block_push_reward:<8.5f} " # Print avg ep block push reward
            f"SPS: {int(global_step/(time.time()-start_time)):<8}")

        if args.track:
            wandb.log({"charts/learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step) # Log to wandb
            wandb.log({"losses/value_loss": v_loss.item()}, step=global_step) # Log to wandb
            wandb.log({"losses/policy_loss": pg_loss.item()}, step=global_step) # Log to wandb
            wandb.log({"losses/entropy": entropy_loss.item()}, step=global_step) # Log to wandb
            wandb.log({"losses/old_approx_kl": old_approx_kl.item()}, step=global_step) # Log to wandb
            wandb.log({"losses/approx_kl": approx_kl.item()}, step=global_step) # Log to wandb
            wandb.log({"losses/clipfrac": np.mean(clipfracs)}, step=global_step) # Log to wandb
            wandb.log({"losses/explained_variance": explained_var}, step=global_step) # Log to wandb
            wandb.log({"charts/SPS": int(global_step / (time.time() - start_time))}, step=global_step) # Log to wandb
            wandb.log({"charts/avg_episodic_return": avg_ep_return}, step=global_step) # Log avg return
            wandb.log({"charts/avg_episodic_length": avg_ep_length}, step=global_step) # Log avg length
            wandb.log({"charts/success_rate": success_rate}, step=global_step) # Log success rate
            wandb.log({"charts/avg_delta_reward": avg_adr}, step=global_step) # Log ADR to wandb
            wandb.log({"charts/avg_episodic_block_push_reward": avg_ep_block_push_reward}, step=global_step) # Log avg ep block push reward


        # Save model checkpoint periodically
        if update % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_update_{update}.pth")
            torch.save(agent.state_dict(), checkpoint_path)
            print(f"Model saved to: {checkpoint_path} at update {update}")


    envs.close()
    if args.track:
        wandb.finish()