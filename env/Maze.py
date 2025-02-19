import math
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import pygame
import os

class BaseMazeEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        "render_fps": 30,
        'video.frames_per_second': 30
    }
    def __init__(self, num_envs=1, maze_layout=None, start_pos=(8.5,3.5),
                 end_pos=(6.5,6.5), step_length=0.2, render_mode=None,
                 device='cuda:0', exploration_reward=0.0): # 添加 exploration_reward 参数
        super().__init__()

        # 公共参数初始化
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.render_mode = render_mode
        self.step_length = step_length
        self.success_threshold = 0.5
        self.exploration_reward = exploration_reward # 保存探索奖励值

        # 初始化迷宫
        if maze_layout is None:
            self.maze = torch.tensor([
                [1,1,1,1,1,1,1,1,1,1,1,1,1],  # Row 0
                [1,0,0,0,0,0,1,0,0,0,0,0,1],  # Row 1
                [1,0,1,1,1,0,1,0,1,1,1,0,1],  # Row 2
                [1,0,1,0,1,0,1,0,1,0,1,0,1],  # Row 3
                [1,0,1,0,0,0,1,0,0,0,1,0,1],  # Row 4
                [1,0,0,1,1,0,0,0,1,1,0,0,1],  # Row 5
                [1,1,0,0,0,0,0,0,0,0,0,1,1],  # Row 6 (包含中心点)
                [1,0,0,1,1,0,0,0,1,1,0,0,1],  # Row 7
                [1,0,1,0,0,0,1,0,0,0,1,0,1],  # Row 8
                [1,0,1,0,1,0,1,0,1,0,1,0,1],  # Row 9
                [1,0,1,1,1,0,1,0,1,1,1,0,1],  # Row 10
                [1,0,0,0,0,0,1,0,0,0,0,0,1],  # Row 11
                [1,1,1,1,1,1,1,1,1,1,1,1,1]   # Row 12
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

        # 成功率追踪初始化
        if self.use_random_start:
            self.success_counts = torch.zeros(len(self.valid_centers), dtype=torch.float32, device=self.device)
            self.episode_counts = torch.ones(len(self.valid_centers), dtype=torch.float32, device=self.device) * 1e-6 # 避免除以零
        else:
            self.success_counts = None
            self.episode_counts = None
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
    def _vectorized_collision_detect(self, new_pos):
        # 修改后的路径获取
        path_coords = self._batch_get_visited_cells(self.current_pos, new_pos)  # 直接返回三维张量

        # 调整维度处理 (batch_size, max_steps, 2)
        grid_coords = path_coords.long()

        # 创建有效掩码（使用逐元素判断）
        valid_mask = (grid_coords[..., 0] >= 0) & (grid_coords[..., 0] < self.cols) & \
                    (grid_coords[..., 1] >= 0) & (grid_coords[..., 1] < self.rows)

        # 获取迷宫值（使用张量索引优化）
        y_coords = torch.clamp(grid_coords[..., 1], 0, self.rows-1)
        x_coords = torch.clamp(grid_coords[..., 0], 0, self.cols-1)
        maze_values = self.maze[y_coords, x_coords]

        # 计算碰撞（路径上有墙或越界）
        collision_mask = (
            (maze_values == 1).any(dim=1) |  # 路径上有墙
            (~valid_mask.all(dim=1))         # 路径越界
        )

        return collision_mask
    # 以下为公共方法
    def reset(self, env_indices=None):
        if env_indices is None:
            env_indices = torch.arange(self.num_envs, device=self.device) # 赋值所有环境索引

        self.steps[env_indices] = 0
        self.returns[env_indices] = 0
        self.visited_matrix[env_indices] = False # 重置指定环境的访问记录


        # 新增随机出生点逻辑
        if self.use_random_start:
            # 计算成功率 (只针对需要重置的环境的起始点)
            success_rates = self.success_counts / self.episode_counts
            success_rates = torch.clamp(success_rates, 0.0, 1.0)

            # 计算权重
            weights = 5.0 - 3.0 * success_rates
            weights = torch.clamp(weights, 2.0, 5.0)

            # 使用 torch.multinomial 进行带权重的随机选择 (只为需要重置的环境选择)
            num_to_reset = len(env_indices)
            indices = torch.multinomial(weights, num_samples=num_to_reset, replacement=True)

            self.current_pos[env_indices] = self.valid_centers[indices]
            self.start_indices[env_indices] = indices # 记录起始点索引 for reset envs
        else:
            # 原有固定起点逻辑
            if isinstance(env_indices, np.ndarray):
                env_indices = torch.from_numpy(env_indices).to(self.device)
            self.current_pos[env_indices] = self.start_pos[env_indices]


        self.dones[env_indices] = False
        return self.current_pos.clone(), {}

    def step(self, actions):
        self.steps += 1  # 更新时间步
        # 动作转换交给子类实现
        action_vectors = self._convert_action(actions)

        with torch.no_grad():
            new_pos = self.current_pos + action_vectors
            collision_mask = self._simple_collision_detect(new_pos)
            success_mask = torch.norm(new_pos - self.end_pos.float(), dim=1) < self.success_threshold
            terminated = collision_mask | success_mask
            truncated = self.steps >= 128
            self.dones = terminated | truncated

            rewards = torch.where(success_mask, 10, torch.where(collision_mask, -1.0, torch.where(truncated,-0.5,0.0))) # 基础奖励

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
            self.current_pos = torch.where(self.dones.unsqueeze(1), self.current_pos, new_pos)

        # 构建info信息（仅返回完成环境的数据）
        final_info = []
        done_indices = torch.where(self.dones)[0]

        # 记录完成环境的信息
        for idx in done_indices:
            info = {
                "steps": self.steps[idx].item(),
                "returns": self.returns[idx].item(),
                "success": success_mask[idx].item(),
                "collision": collision_mask[idx].item()
            }
            final_info.append(info)
            if self.use_random_start:
                start_index = self.start_indices[idx] # 获取起始点索引
                self.episode_counts[start_index] += 1 # 增加 episode 计数
                if success_mask[idx]:
                    self.success_counts[start_index] += 1 # 成功时增加成功计数


        # 自动重置完成的环境
        if torch.any(self.dones):
            reset_indices = done_indices.clone()
            self.reset(reset_indices)

        self.render()

        if self.use_random_start and torch.sum(self.episode_counts) > 0:
            average_success_rate = torch.mean(self.success_counts / self.episode_counts)
        else:
            average_success_rate = torch.tensor(float('nan'), device=self.device) # 如果不使用随机起始点，则返回 NaN

        return (
            self.current_pos.clone(),
            rewards,
            terminated,
            truncated,
            {"final_info": final_info,
             "MSR": average_success_rate} if final_info else {}  # 仅当有完成环境时返回
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
        if self.use_random_start and self.success_counts is not None and self.episode_counts is not None:
            for i in range(len(self.valid_centers)):
                cell_center = self.valid_centers[i]
                cell_x = int(cell_center[0] - 0.5)
                cell_y = int(cell_center[1] - 0.5)
                success_rate = self.success_counts[i] / self.episode_counts[i]
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
        self.action_space = spaces.Box(
            low=-self.step_length,
            high=self.step_length,
            shape=(2,),
            dtype=np.float32
        )

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


# 使用示例
if __name__ == "__main__":
    # 配置参数
    num_envs = 4  # 并行环境数量
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 直接创建环境实例
    # env = ContinuousMaze(
    env = Discrete8Maze(
        num_envs=num_envs,
        render_mode='human',
        device=device,
        start_pos=None,
        exploration_reward = 0.0 # 设置探索奖励为 0.5
    )

    # 训练循环
    obs, _ = env.reset()
    for step in range(10000):
        # 生成动作
        with torch.no_grad():
            actions = env.action_space.sample()
            # actions = torch.zeros((num_envs,2), device=device)
            # actions[:,1]=1

        # 环境交互
        obs, rewards, dones, _, _ = env.step(actions)
        obs = torch.as_tensor(obs, device=device)


        # print(f'Step {step},Done: {dones.cpu().item()},Obs: {obs.cpu().numpy()},Reward: {rewards.cpu().numpy()}')
        # if env.render_mode == 'human': # 只有在 human 模式下才渲染，否则会影响速度
        #     env.render()

    env.close()