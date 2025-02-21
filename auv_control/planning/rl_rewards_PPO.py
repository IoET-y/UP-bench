import numpy as np
def calculate_reward(self, state, next_state, action):

    """
    优化后的奖励函数，每步奖励经过缩放后保证整个 episode（最多 1000 步）
    累计奖励大致在 [-1, 1] 内，同时各项目标明确，有助于 PPO 快速收敛。
    """

    max_steps = 1000.0
    norm_factor = 1.0 / max_steps  # 每步奖励缩放因子

    # 假设已知初始距离（可在环境初始化时记录）
    # 若没有，可以用当前距离做近似；这里假定 self.start 已定义
    D_max = 1# np.linalg.norm(self.start - self.end) if hasattr(self, 'start') else 1.0

    # 1. 进展奖励 + 势函数 shaping（采用潜力函数 phi = -distance / D_max）
    old_distance = np.linalg.norm(state[0:3] - self.end)
    new_distance = np.linalg.norm(next_state[0:3] - self.end)
    progress = old_distance - new_distance

    phi_old = -old_distance / D_max
    phi_new = -new_distance / D_max
    shaping_reward = phi_new - phi_old  # 自然反映了距离减少的比例

    # 若长时间停滞则给予额外惩罚
    if progress < 0.01:
        self.static_counter += 1
    else:
        self.static_counter = 0
    stagnation_penalty = -0.5 if self.static_counter >= 50 else 0

    # 2. 对齐奖励：鼓励运动方向与目标方向一致
    displacement = next_state[0:3] - state[0:3]
    if np.linalg.norm(displacement) > 1e-6:
        displacement_direction = displacement / np.linalg.norm(displacement)
    else:
        displacement_direction = np.zeros_like(displacement)
    goal_direction = self.end - state[0:3]
    if np.linalg.norm(goal_direction) > 1e-6:
        goal_direction /= np.linalg.norm(goal_direction)
    else:
        goal_direction = np.zeros_like(goal_direction)
    alignment = np.dot(displacement_direction, goal_direction)
    # 这里用进展占总距离的比例作为权重，保证量纲一致
    alignment_reward = 0.05 * alignment * (progress / D_max)

    # 3. 安全惩罚：避免接近障碍
    distances_to_obstacles = state[6:20]
    min_distance = np.min(distances_to_obstacles)
    collision_threshold = 1.0   # 碰撞临界距离
    safe_distance_threshold = 1.5  # 理想安全距离
    if min_distance < collision_threshold:
        safety_penalty = -1.0  # 碰撞时一次性较大惩罚
        self.episode_collisions += 1
    elif min_distance < safe_distance_threshold:
        # 线性惩罚，惩罚幅度在 0 到 -0.5 之间
        safety_penalty = -0.5 * (safe_distance_threshold - min_distance) / (safe_distance_threshold - collision_threshold)
    else:
        safety_penalty = 0.0

    # 4. 能耗惩罚：鼓励小幅控制，防止大幅动作
    energy_penalty = -0.01 * (np.linalg.norm(action) ** 2)

    # 5. 平滑性惩罚：鼓励连续动作平滑变化
    smoothness_penalty = -0.01 * np.linalg.norm(action - self.previous_action)

    # 6. 时间惩罚：每步都给予一个极小的负奖励，促使尽快到达终点
    time_penalty = -0.001

    # 7. 达标奖励：靠近目标时获得额外奖励，设计为目标距离越近奖励越高（但最大不超过 1）
    target_threshold = 2.0
    bonus = 0.0
    if new_distance < target_threshold:
        bonus = 1  # 在 [0, 1] 内

    # 8. 越界惩罚：保证状态在合理区域内
    lower_bound = np.array([-1, -1, -100])
    upper_bound = np.array([100, 100, 0])
    if np.any(next_state[0:3] < lower_bound) or np.any(next_state[0:3] > upper_bound):
        out_of_box_penalty = -0.1
    else:
        out_of_box_penalty = 0.0

    # 汇总各项奖励（未缩放前）
    step_reward = (shaping_reward +
                   stagnation_penalty +
                   alignment_reward +
                   safety_penalty +
                   energy_penalty +
                   smoothness_penalty +
                   time_penalty +
                   bonus +
                   out_of_box_penalty)

    # 缩放奖励，使得最多 1000 步累积奖励接近 [−1, 1]
    normalized_step_reward = step_reward * norm_factor

    # 更新记录指标（可根据需要保留）
    self.total_length += abs(progress)
    self.episode_align_reward += alignment_reward
    self.episode_safety_reward += safety_penalty
    self.episode_energy += np.linalg.norm(action) ** 2
    self.episode_smoothness += np.linalg.norm(action - self.previous_action)
    step_progress = np.linalg.norm(next_state[0:3] - state[0:3])
    self.episode_path_length += step_progress
    self.episode_out_of_box_penalty += out_of_box_penalty
    self.episode_energy_penalty += energy_penalty
    self.episode_smoothness_penalty += smoothness_penalty
    self.episode_time_penalty += time_penalty
    if new_distance < target_threshold:
        self.episode_reach_target_reward += bonus

    # 更新上一动作
    self.previous_action = action.copy()

    return normalized_step_reward, shaping_reward, alignment_reward, safety_penalty, bonus, smoothness_penalty