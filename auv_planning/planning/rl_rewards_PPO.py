import numpy as np

def calculate_reward(self, state, next_state, action):
    """
    优化后的奖励函数：
      - 对于连续奖励部分（进展、对齐、安全、能耗、平滑、时间惩罚）使用统一缩放因子（1/1000），
        使得多步累积奖励较小；
      - 对于事件奖励（达标奖励、越界惩罚），只在触发时一次性给予，不进行缩放。
    """
    max_steps = 1000.0
    norm_factor = 1.0 / max_steps  # 连续奖励缩放因子

    # 计算最大距离（起点到终点），保证尺度统一
    D_max = np.linalg.norm(self.start - self.end) if hasattr(self, 'start') else 1.0
    state[0:3] = state[0:3] * 10
    next_state[0:3] = next_state[0:3] * 10
    # 1. 进展奖励 + 势函数 shaping（使用潜力函数 phi = -distance / D_max）
    old_distance = np.linalg.norm(state[0:3] - self.end)
    new_distance = np.linalg.norm(next_state[0:3] - self.end)
    progress = old_distance - new_distance
    phi_old = -old_distance / D_max
    phi_new = -new_distance / D_max
    shaping_reward = phi_new - phi_old  # 若向目标靠近，则为正

    # 2. 停滞惩罚：若进展非常小，则累计停滞计数，超过一定步数后给予惩罚
    if progress < 0.01:
        self.static_counter += 1
    else:
        self.static_counter = 0
    stagnation_penalty = -0.5 if self.static_counter >= 50 else 0

    # 3. 对齐奖励：鼓励运动方向与目标方向一致
    displacement = next_state[0:3] - state[0:3]
    if np.linalg.norm(displacement) > 1e-6:
        displacement_direction = displacement / np.linalg.norm(displacement)
    else:
        displacement_direction = np.zeros_like(displacement)
    goal_direction = self.end - state[0:3]
    if np.linalg.norm(goal_direction) > 1e-6:
        goal_direction = goal_direction / np.linalg.norm(goal_direction)
    else:
        goal_direction = np.zeros_like(goal_direction)
    alignment = np.dot(displacement_direction, goal_direction)
    # 调低对齐奖励的系数，从原来的 50 降至 10
    alignment_reward = 50 * alignment * (abs(progress) / D_max)

    # 4. 安全惩罚：根据与障碍物的最小距离给予惩罚
    distances_to_obstacles = state[6:20]*10
    min_distance = np.min(distances_to_obstacles)
    collision_threshold = 1.0    # 碰撞临界距离
    safe_distance_threshold = 1.5  # 理想安全距离
    if min_distance < collision_threshold:
        safety_penalty = -1.0
        self.episode_collisions += 1
    elif min_distance < safe_distance_threshold:
        safety_penalty = -0.5 * (safe_distance_threshold - min_distance) / (safe_distance_threshold - collision_threshold)
    else:
        safety_penalty = 0.0

    # 5. 能耗惩罚：鼓励小幅控制
    energy_penalty = -0.05 * (np.linalg.norm(action) ** 2)

    # 6. 平滑性惩罚：鼓励连续动作平滑变化
    smoothness_penalty = -0.05 * np.linalg.norm(action - self.previous_action)

    # 7. 时间惩罚：每步给予一个小的负奖励，促使尽快到达目标
    time_penalty = -0.005

    # 连续奖励部分（将上述各项累加后统一缩放）
    factor = 0
    continuous_reward = (shaping_reward * factor +
                         stagnation_penalty * factor +
                         alignment_reward +
                         safety_penalty  +
                         energy_penalty * factor +
                         smoothness_penalty * factor +
                         time_penalty * factor)
    scaled_continuous_reward = continuous_reward * norm_factor

    # 8. 达标奖励（事件奖励）：当接近目标时给予一次性奖励（这里不进行缩放）
    target_threshold = 2.0
    bonus = 0.0
    if new_distance < target_threshold:
        bonus = 1.0  # 达到目标时奖励 1.0（可视情况调整，并建议结束当前 episode）

    # 9. 越界惩罚（事件惩罚）：若状态超出预定范围，给予一次性惩罚
    lower_bound = np.array([-5, -5, -100])
    upper_bound = np.array([100, 100, 0])
    if np.any(next_state[0:3] < lower_bound) or np.any(next_state[0:3] > upper_bound):
        out_of_box_penalty = -0.1  # 超出范围时一次性扣分（可考虑立即结束 episode）
    else:
        out_of_box_penalty = 0.0

    # 最终奖励 = 缩放后的连续奖励 + 事件奖励
    step_reward = scaled_continuous_reward + bonus + out_of_box_penalty

    # 更新统计指标（可根据需要记录）
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

    # 更新上一动作记录
    self.previous_action = action.copy()

    return step_reward, shaping_reward, alignment_reward, safety_penalty, bonus, smoothness_penalty