import numpy as np

def calculate_reward(self, state, next_state, action):
    """
    改进后的奖励函数，包含：
    - 进展奖励：基于目标距离减少量和势函数 shaping
    - 对齐奖励：鼓励运动方向与目标方向一致
    - 安全惩罚：采用平滑的连续函数惩罚靠近障碍物
    - 能耗惩罚：增加权重以减少大幅控制
    - 平滑性惩罚：鼓励连续动作平滑变化
    - 时间惩罚：固定扣分
    - 达标奖励：靠近目标时获得额外奖励
    """

    # 参数权重（可调）
    factor = 0.0
    w_progress   = 0
    w_shaping    = 0.50   # 势函数 shaping 权重
    w_alignment  = 3  # 5 for small ，3 for 300* 300 * 100
    w_safety     = 2  * factor # 安全惩罚较大，鼓励避障
    w_energy     = 0.1 * factor
    w_smoothness = 0.1 * factor
    w_time       = 0.1 * factor
    w_bonus      = 2.5

    gamma = 0.99  # 用于势函数 shaping

    # 1. 进展奖励和势函数 shaping
    old_distance = np.linalg.norm(state[0:3] - self.end)
    new_distance = np.linalg.norm(next_state[0:3] - self.end)
    progress = (old_distance - new_distance)
    if progress < 0.01:
        self.static_counter += 1
    else:
        self.static_counter = 0
    progress_reward = -2 if self.static_counter >= 50 else 0


    # 2. 对齐奖励
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
    alignment_reward = w_alignment * alignment * abs(progress) if progress > 0.0001 else 0 # 根据进展幅度加权

    # 3. 安全惩罚（连续函数形式）
    distances_to_obstacles = state[6:20]
    min_distance = np.min(distances_to_obstacles)
    # 使用 sigmoid 函数模拟，当距离较小时惩罚迅速上升
    safety_penalty = 0.0
    collision_threshold = 1  # 进入碰撞区域
    safe_distance_threshold = 1.5  # 理想安全距离

    if min_distance < collision_threshold:
        safety_penalty = -w_safety * 100.0  # 碰撞处大惩罚
        self.episode_collisions += 1
    else:
        # 当处于 [collision_threshold, safe_distance_threshold] 之间时，惩罚平滑上升
        if min_distance < safe_distance_threshold:
            # 可采用线性或指数函数，此处采用简单线性
            safety_penalty = -w_safety * (safe_distance_threshold - min_distance)

    # 4. 能耗惩罚
    energy_penalty = -w_energy * (np.linalg.norm(action) ** 2)

    # 5. 平滑性惩罚
    smoothness_penalty = -w_smoothness * np.linalg.norm(action - self.previous_action)

    # 6. 时间惩罚
    time_penalty = -w_time

    # 7. 达标奖励：靠近目标时给予额外奖励，可设计为连续奖励
    bonus = 0.0
    target_threshold = 2.0
    if new_distance < target_threshold:
        # 当距离目标越近，奖励可以越大（例如线性或指数）
        bonus = 500


#8 如越界，给予额外惩罚
    lower_bound = np.array([-1, -1, -100])
    upper_bound = np.array([100, 100, 0])


    if np.any(next_state[0:3] < lower_bound) or np.any(next_state[0:3] > upper_bound) :
        out_of_box_penalty = - 10
    else:
        out_of_box_penalty = 0

    # 汇总各项奖励
    total_reward = (progress_reward  +
                    alignment_reward + safety_penalty +
                    energy_penalty + smoothness_penalty +
                    time_penalty + bonus + out_of_box_penalty)

    # 更新记录指标
    self.total_length += abs(progress)
    self.episode_align_reward += alignment_reward
    self.episode_safety_reward += safety_penalty
    self.episode_energy += np.linalg.norm(action) ** 2
    self.episode_smoothness += np.linalg.norm(action - self.previous_action)
    step_progress = np.linalg.norm(next_state[0:3] - state[0:3])
    self.episode_path_length += step_progress
    self.episode_out_of_box_penalty += out_of_box_penalty
    self.episode_energy_penalty +=  energy_penalty
    self.episode_smoothness_penalty += smoothness_penalty
    self.episode_time_penalty += time_penalty
    if new_distance < target_threshold:
        self.episode_reach_target_reward += bonus

    # 更新上一动作
    self.previous_action = action.copy()

    return total_reward, progress_reward, alignment_reward, safety_penalty, bonus, smoothness_penalty