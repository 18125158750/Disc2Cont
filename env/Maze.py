import math
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import pygame
import os
from collections import deque

class BaseMazeEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        "render_fps": 30,
        'video.frames_per_second': 30
    }
    def __init__(self, num_envs=1, maze_layout=None, start_pos=(8.5,3.5),
                 end_pos=(9.5,8.5), step_length=0.2, render_mode=None,
                 device='cuda:0', exploration_reward=0.0, action_repeat_probability=0.0,
                 timeout=128, success_rate_influence=0.4):
        super().__init__()

        if not hasattr(self, 'action_space'):
            self.action_space = spaces.Box(
                low=-step_length,
                high=step_length,
                shape=(2,),
                dtype=np.float32
            )

        # 公共参数初始化
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.render_mode = render_mode
        self.step_length = step_length
        self.success_threshold = 0.5
        self.exploration_reward = exploration_reward # 保存探索奖励值
        self.action_repeat_probability = action_repeat_probability # 保存动作重复概率
        self.timeout = timeout
        self.success_rate_influence = success_rate_influence

        # 初始化迷宫
        if maze_layout is None:
            self.maze = torch.tensor([
                [1,1,1,1,1,1,1,1,1,1,1],
                [1,0,1,1,1,1,0,0,0,0,1],
                [1,0,0,0,1,0,0,1,1,0,1],
                [1,1,1,0,1,0,1,1,0,0,1],
                [1,1,0,0,0,0,1,1,1,1,1],
                [1,0,0,1,1,0,1,0,0,0,1],
                [1,0,1,1,0,0,0,0,1,0,1],
                [1,0,0,1,0,1,1,0,1,1,1],
                [1,1,1,1,0,0,1,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,1,1]
            ], device=self.device)
        else:
            self.maze = torch.tensor(maze_layout, device=self.device)

        # 公共状态初始化
        self._init_positions(start_pos, end_pos)
        self._init_rendering()
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.array([self.cols, self.rows], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.recording = False
        # 状态存储
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        # 时间步追踪
        self.steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        # 奖励追踪
        self.returns = torch.zeros(num_envs, dtype=torch.int32, device=device)
        # 全过程碰撞追踪
        self.episode_collision = torch.zeros(num_envs, dtype=torch.bool, device=self.device) # 初始化全过程碰撞记录
        # 上一次动作存储
        self.last_actions = torch.zeros((num_envs,) + self.action_space.shape, dtype=torch.float32, device=self.device) # 初始化上一次动作为 0

        # 成功率追踪初始化
        if self.use_random_start:
            self.success_history_deques = [deque(maxlen=1000) for _ in range(len(self.valid_centers))]
            self._cached_success_rates = torch.zeros(len(self.valid_centers), dtype=torch.float32, device=self.device) # 使用 Tensor 缓存每个起始点的成功率
            self._cached_average_success_rate = torch.tensor(0.5, dtype=torch.float32, device=self.device) # 使用 Tensor 缓存平均成功率
            self._cache_update_interval = 1000 # 每隔多少步更新一次缓存 (设置为 1000)
            self._cache_update_counter = 0
        else:
            self.success_history_deques = None
            self._cached_success_rates = None
            self._cached_average_success_rate = None
            self._cache_update_interval = 0
            self._cache_update_counter = 0
        self.start_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device) # 记录每个环境的起始点索引

    def _init_positions(self, start_pos, end_pos):
        """初始化起点终点公共逻辑"""
        self.rows, self.cols = self.maze.shape
        valid_y, valid_x = torch.where(self.maze == 0)
        self.valid_centers = torch.stack([
            valid_x.float() + 0.5, valid_y.float() + 0.5
        ], dim=1).to(self.device)

        # 创建 grid_index_map
        self.grid_index_map = torch.full((self.rows, self.cols), -1, dtype=torch.long, device=self.device) # 初始化为 -1
        for i in range(len(self.valid_centers)):
            center = self.valid_centers[i]
            cell_x = int(center[0] - 0.5)
            cell_y = int(center[1] - 0.5)
            self.grid_index_map[cell_y, cell_x] = i # 使用 cell_y, cell_x 作为索引


        # 过滤终点坐标
        end_pos_tensor = torch.tensor(end_pos, device=self.device).unsqueeze(0)
        is_end = torch.all(torch.abs(self.valid_centers - end_pos_tensor) < 1e-3, dim=1)
        self.valid_centers = self.valid_centers[~is_end]


        # 初始位置设置
        self.use_random_start = (start_pos is None)
        self.start_pos = self._get_start_pos(start_pos)
        self.end_pos = torch.tensor(end_pos, device=self.device)
        self.end_pos_np = np.array(end_pos)
        self.current_pos = self.start_pos.clone()
        self.initial_start_pos = self.start_pos.clone() # 保存初始起点位置

        # 探索奖励相关初始化
        self.visited_matrix = torch.zeros((self.num_envs, len(self.valid_centers)), dtype=torch.bool, device=self.device) # 访问记录矩阵

    def _get_start_pos(self, start_pos):
        """获取起点位置公共方法"""
        if self.use_random_start:
            indices = torch.randint(0, len(self.valid_centers),
                                  (self.num_envs,), device=self.device)
            return self.valid_centers[indices]
        return torch.tensor(start_pos, device=self.device).repeat(self.num_envs, 1)

    def _init_rendering(self):
        """渲染初始化公共逻辑"""
        self.cell_size = 64
        self.window_size = (self.cols * self.cell_size, self.rows * self.cell_size)
        self.clock = pygame.time.Clock()
        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
        else:
            self.window = None

    # 以下为需要子类实现的抽象方法
    def _get_action_space(self):
        raise NotImplementedError

    def _convert_action(self, actions):
        raise NotImplementedError

    def _simple_collision_detect(self, new_pos):
        """简化版碰撞检测：仅检查终点所在网格是否碰撞"""
        # 转换为网格坐标
        grid_x = torch.floor(new_pos[:, 0]).long()
        grid_y = torch.floor(new_pos[:, 1]).long()

        # 边界约束
        valid_x = torch.clamp(grid_x, 0, self.cols-1)
        valid_y = torch.clamp(grid_y, 0, self.rows-1)

        # 获取迷宫值
        maze_values = self.maze[valid_y, valid_x]

        # 碰撞条件（所在网格是墙 或 越界）
        out_of_bounds = (new_pos[:, 0] < 0) | (new_pos[:, 0] >= self.cols) | \
                        (new_pos[:, 1] < 0) | (new_pos[:, 1] >= self.rows)

        return (maze_values == 1) | out_of_bounds

    # 以下为公共方法
    def reset(self, env_indices=None):
        if env_indices is None:
            env_indices = torch.arange(self.num_envs, device=self.device) # 赋值所有环境索引

        self.steps[env_indices] = 0
        self.returns[env_indices] = 0
        self.visited_matrix[env_indices] = False # 重置指定环境的访问记录
        self.episode_collision[env_indices] = False # 重置全过程碰撞记录
        self.last_actions[env_indices] = 0.0 # 重置last_actions

        # 新增随机出生点逻辑
        if self.use_random_start:
            # 直接使用缓存的平均成功率，不再计算
            success_rates = self._cached_success_rates
            success_rates = torch.clamp(success_rates, 0.0, 1.0)

            # 计算权重
            weights = 1.0 - self.success_rate_influence * success_rates
            # weights = torch.ones_like(success_rates)
            weights = torch.clamp(weights, (1.0 - self.success_rate_influence), 1.0)

            # 使用 torch.multinomial 进行带权重的随机选择 (只为需要重置的环境选择)
            num_to_reset = len(env_indices)
            indices = torch.multinomial(weights, num_samples=num_to_reset, replacement=True)

            self.current_pos[env_indices] = self.valid_centers[indices]
            self.start_indices[env_indices] = indices # 记录起始点索引 for reset envs
            self.initial_start_pos[env_indices] = self.current_pos[env_indices].clone() # 更新初始起点位置
        else:
            # 原有固定起点逻辑
            if isinstance(env_indices, np.ndarray):
                env_indices = torch.from_numpy(env_indices).to(self.device)
            self.current_pos[env_indices] = self.start_pos[env_indices]
            self.initial_start_pos[env_indices] = self.start_pos[env_indices].clone() # 更新初始起点位置


        self.dones[env_indices] = False
        return self.current_pos.clone(), {}

    def step(self, actions):
        self.steps += 1  # 更新时间步

        # 动作转换交给子类实现
        action_vectors = self._convert_action(actions)

        # 动作重复逻辑
        repeat_action_mask = (torch.rand(self.num_envs, device=self.device) < self.action_repeat_probability)
        selected_actions = torch.where(repeat_action_mask.unsqueeze(-1), self.last_actions, action_vectors) # 选择重复动作或新动作

        with torch.no_grad():
            new_pos = self.current_pos + action_vectors
            collision_mask = self._simple_collision_detect(new_pos)
            success_mask = torch.norm(new_pos - self.end_pos.float(), dim=1) < self.success_threshold
            terminated = success_mask # 碰撞不再终止 episode, 只有成功才终止
            truncated = self.steps >= self.timeout
            self.dones = terminated | truncated

            rewards = torch.where(success_mask, 10, torch.where(collision_mask, -1.0, torch.where(truncated,0.0,0.0))) # 基础奖励

            if self.exploration_reward>0:
                # 计算探索奖励 (向量化)
                grid_x = torch.floor(new_pos[:, 0]).long()
                grid_y = torch.floor(new_pos[:, 1]).long()

                # 使用 grid_index_map 直接获取索引 (向量化)
                cell_indices = self.grid_index_map[grid_y, grid_x]

                # 计算终点格子的网格坐标
                end_cell_x = int(self.end_pos_np[0] - 0.5) # 使用 numpy 版本的 end_pos 方便计算
                end_cell_y = int(self.end_pos_np[1] - 0.5)

                # 创建终点格子掩码
                is_end_cell_mask = ~((grid_x == end_cell_x) & (grid_y == end_cell_y)) # 不是终点格子的为 True

                # 排除无效索引 (-1) 的格子，只考虑有效格子
                invalid_index_mask = (cell_indices != -1)
                valid_cell_mask = invalid_index_mask & is_end_cell_mask # **同时满足两个条件才是有效的格子**
                valid_cell_indices = cell_indices[valid_cell_mask]

                # 使用矩阵访问记录来判断是否是新格子 (仅针对有效格子)
                is_new_cell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # 默认都是 False
                if (valid_cell_indices>=40).any():
                    print(valid_cell_indices)
                is_new_cell[valid_cell_mask] = ~self.visited_matrix[valid_cell_mask, valid_cell_indices] # 只更新有效格子部分

                # 计算探索奖励，只有新格子才有奖励
                exploration_rewards = is_new_cell * self.exploration_reward

                # 更新访问记录矩阵 (仅针对有效格子)
                self.visited_matrix[valid_cell_mask, valid_cell_indices] = True


                rewards += exploration_rewards # 将探索奖励加到总奖励中
            self.returns = self.returns + rewards

            # 更新全过程碰撞记录 (使用逻辑 OR)
            self.episode_collision = self.episode_collision | collision_mask
            # 更新 last_actions
            self.last_actions = selected_actions

            # 碰撞后返回当前环境的起点，否则更新位置
            self.current_pos = torch.where(collision_mask.unsqueeze(1), self.initial_start_pos, new_pos)
            self.current_pos = torch.where(self.dones.unsqueeze(1), self.current_pos, self.current_pos) # 保持done状态的位置不变，防止重置位置在done时发生


        # 构建info信息（仅返回完成环境的数据）
        final_info = []
        done_indices = torch.where(self.dones)[0]

        # 计算 MASR (各个起始点权重相等)
        if self.use_random_start:
            current_batch_successes = success_mask[done_indices]
            current_episode_collisions = self.episode_collision[done_indices] # 获取完成episode的全程碰撞信息
            current_start_indices = self.start_indices[done_indices] # 获取完成环境的起始点索引

            for i in range(len(done_indices)): # 遍历完成的环境
                start_index = current_start_indices[i].item()
                success = current_batch_successes[i].item()
                episode_collision = current_episode_collisions[i].item()
                episode_success_no_collision = success and not episode_collision # 新的成功条件
                self.success_history_deques[start_index].append(episode_success_no_collision) # 添加到对应起始点的 deque


            # 定期计算成功率
            if self._cache_update_counter % self._cache_update_interval == 0:
                success_rates_list_np = [] # 使用 numpy list 临时存储，然后转 tensor
                for i, deque_ in enumerate(self.success_history_deques):
                    if deque_:
                        success_rates_list_np.append(np.mean(deque_))
                    else:
                        success_rates_list_np.append(0.0)
                if success_rates_list_np:
                    self._cached_success_rates = torch.tensor(success_rates_list_np, dtype=torch.float32, device=self.device) # 更新缓存的每个起始点成功率
                    self._cached_average_success_rate = torch.mean(self._cached_success_rates)
                else:
                    self._cached_average_success_rate = torch.tensor(0.5, dtype=torch.float32, device=self.device)


            masr = self._cached_average_success_rate.item() # 直接使用缓存的平均成功率
        else:
            masr = float('nan')


        # 记录完成环境的信息
        for idx in done_indices:
            info = {
                "steps": self.steps[idx].item(),
                "returns": self.returns[idx].item(),
                "success": success_mask[idx].item(),
                "collision": self.episode_collision[idx].item() # 添加全过程碰撞信息
            }
            final_info.append(info)



        # 自动重置完成的环境
        if torch.any(self.dones):
            reset_indices = done_indices[self.dones[done_indices]].clone() # 仅对done=True的环境重置
            self.reset(reset_indices)

        self.render()
        self._cache_update_counter += 1 # 增加计数器


        return (
            self.current_pos.clone(),
            rewards,
            terminated,
            truncated,
            {"final_info": final_info,
             "MASR": masr} if final_info else {}  # 仅当有完成环境时返回
        )


    def render(self):
        if self.render_mode != "human" and not self.recording:
            return

        # 只渲染第一个环境
        pos = self.current_pos[0].cpu().numpy()

        # 渲染代码与之前类似，增加窗口事件处理
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

        # 颜色定义
        colors = {
            "background": (255, 255, 255),
            "wall": (40, 40, 40),
            "agent": (255, 0, 0),
            "goal": (255, 255, 0), # 目标颜色改为黄色
            "path": (200, 200, 200),
        }

        # 创建画布
        if self.render_mode == "human":
            canvas = self.window
        else:
            canvas = pygame.Surface(self.window_size)

        canvas.fill(colors["background"])

        # 绘制迷宫
        for y in range(self.rows):
            for x in range(self.cols):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(canvas, colors["wall"], rect)

        # 绘制成功率方格
        if self.use_random_start and self.success_history_deques is not None:
            for i in range(len(self.valid_centers)):
                cell_center = self.valid_centers[i]
                cell_x = int(cell_center[0] - 0.5)
                cell_y = int(cell_center[1] - 0.5)

                success_rate = self._cached_success_rates[i].item() # 使用缓存的成功率

                green_intensity = int(255-success_rate * 255)
                green_intensity = max(0, min(green_intensity, 255)) # 确保在 0-255 范围内
                success_color = (green_intensity, 255, green_intensity) # Green, darker for higher success rate


                rect = pygame.Rect(
                    cell_x * self.cell_size,
                    cell_y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(canvas, success_color, rect)


        # 绘制终点 (黄色方格)
        goal_rect = pygame.Rect(
            int(self.end_pos_np[0] - 0.5) * self.cell_size, # 调整起始坐标
            int(self.end_pos_np[1] - 0.5) * self.cell_size, # 调整起始坐标
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(canvas, colors["goal"], goal_rect)


        # 绘制机器人
        agent_pos = (
            int(pos[0] * self.cell_size),
            int(pos[1] * self.cell_size)
        )
        pygame.draw.circle(canvas, colors["agent"], agent_pos, 8)

        # 视频录制
        if self.recording:
            # 获取当前帧（只记录第一个环境）
            frame = np.transpose(
                pygame.surfarray.pixels3d(canvas),
                axes=(1, 0, 2)
            )
            self.video_frames.append(frame.copy())

            # 自动停止逻辑
            if len(self.video_frames) >= 1000:
                self.stop_recording()


        # 更新显示
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def start_recording(self, video_path):
        """启动录制，传入视频保存路径"""
        if not self.recording:
            self.recording = True
            self.video_frames = []
            self.video_path = video_path
            os.makedirs(os.path.dirname(video_path), exist_ok=True)


    def stop_recording(self):
        """停止并保存视频"""
        if self.recording:
            import imageio
            # 只保留最多1000帧
            frames = self.video_frames[:1000]
            imageio.mimsave(
                self.video_path,
                frames,
                fps=self.metadata['video.frames_per_second']
            )
            self.recording = False
            self.video_frames = []
            print(f"Video saved to {self.video_path}")

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class ContinuousMaze(BaseMazeEnv):
    """连续动作环境"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _convert_action(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device)*self.step_length

        action_norm = torch.norm(actions, dim=-1, keepdim=True) # 计算二范数，保持维度以便广播
        mask = action_norm > self.step_length # 创建一个掩码，标记哪些动作的范数过大

        # 计算缩放比例，对于范数过大的动作进行缩放
        scaling_factor = torch.ones_like(action_norm) # 初始化缩放因子为 1
        scaling_factor[mask] = self.step_length / action_norm[mask] # 对范数过大的动作计算缩放因子

        actions = actions * scaling_factor # 应用缩放因子
        return actions

class Discrete4Maze(BaseMazeEnv):
    """四方向离散动作环境"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(4)
        self._directions = torch.tensor([
            [0, -1], [0, 1], [-1, 0], [1, 0]
        ], device=self.device) * self.step_length

    def _convert_action(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        return self._directions[actions]

class Discrete8Maze(BaseMazeEnv):
    """八方向离散动作环境"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(8)
        diag_step = math.sqrt(0.5)
        self._directions = torch.tensor([
            [0, -1], [0, 1], [-1, 0], [1, 0],
            [-diag_step, -diag_step], [diag_step, -diag_step],
            [-diag_step, diag_step], [diag_step, diag_step]
        ], device=self.device) * self.step_length

    def _convert_action(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        return self._directions[actions]
    
class Discrete16Maze(BaseMazeEnv):
    """十六方向离散动作环境"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(16)
        directions_list = []
        for i in range(16):
            angle = 2 * math.pi * i / 16
            directions_list.append([math.cos(angle), math.sin(angle)])
        self._directions = torch.tensor(directions_list, device=self.device) * self.step_length

    def _convert_action(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        return self._directions[actions]
    
class DiscreteRandomActionMaze(BaseMazeEnv):
    """16选8+1方向离散动作环境"""
    # 每一次action之后会随机切换动作
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(8+1)
        self.directions_list = []
        for i in range(16):
            angle = 2 * math.pi * i / 16
            self.directions_list.append([math.cos(angle), math.sin(angle)])
        self.directions_list.append([0, 0])
        # 从directions_list的前16个动作中任选一个加上最后一个动作作为self._directions
        # 动作的步长范围为[0.5*self.step_length, self.step_length]
        self._directions = torch.tensor(self.directions_list, device=self.device) * self.step_length

    def _convert_action(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)

        # 采样
        actions_sample = self._directions[actions]
        # 重置方向
        self._set_random_directions()
        return actions_sample

    def _set_random_directions(self):
        """随机选择前16个方向,并设置到self._directions中"""
        random_indices = torch.randperm(16, device=self.device)[:8]
        selected_directions = [self.directions_list[i] for i in random_indices.tolist()]
        selected_directions.append(self.directions_list[-1])
        random_scales = (torch.rand(8, device=self.device) * 0.5 + 0.5).unsqueeze(-1) # 变成(16,1)
        self._directions = torch.tensor(selected_directions, device=self.device) * self.step_length * random_scales

# 使用示例
if __name__ == "__main__":
    # 配置参数
    num_envs = 4  # 并行环境数量
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 直接创建环境实例
    # env = ContinuousMaze(
    env = DiscreteRandomActionMaze(
        num_envs=num_envs,
        render_mode='human',
        device=device,
        start_pos=None,
        exploration_reward = 0.0, # 设置探索奖励为 0.0
        action_repeat_probability = 0.2 # 设置动作重复概率为 20%
    )

    # 训练循环
    obs, _ = env.reset()
    for step in range(10000):
        # 生成动作
        with torch.no_grad():
            # actions = env.action_space.sample()
            # actions = torch.zeros((num_envs,2), device=device)
            actions = torch.zeros((num_envs,), device=device, dtype=torch.long)
            actions[:]=0

        # 环境交互
        obs, rewards, dones, _, info = env.step(actions)
        obs = torch.as_tensor(obs, device=device)

        if info.get('final_info'):
            for env_info in info['final_info']:
                print(f"Episode Final Info: Steps: {env_info['steps']}, Returns: {env_info['returns']}, Success: {env_info['success']}, Episode Collision: {env_info['collision']}")

    env.close()