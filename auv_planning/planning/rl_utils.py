# rl_utils.py
import numpy as np
import random
import pickle

def calculate_ocean_current(self, position, time):
    """
    根据环境配置计算洋流的三维速度。
    如果配置中指定了固定方向和速度，则直接返回固定向量；
    否则采用默认的简单振荡模型。
    """
    level_config = self.config["environment"]["levels"][str(self.env_level)]
    if "current" in level_config:
        current_config = level_config["current"]
        if "speed" in current_config and "direction" in current_config:
            speed = current_config["speed"]
            direction = np.array(current_config["direction"])
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            return speed * direction
    A = self.current_strength
    mu = self.current_mu
    omega = self.current_omega
    a_t = mu * np.sin(omega * time)
    b_t = 1 - 2 * mu * np.sin(omega * time)
    f_x_t = a_t * position[0]**2 + b_t * position[0]
    v_x = -self.current_frequency * A * np.sin(self.current_frequency * f_x_t) * np.cos(self.current_frequency * position[1])
    v_y = self.current_frequency * A * np.cos(self.current_frequency * f_x_t) * np.sin(self.current_frequency * position[1])
    v_z = 0
    return np.array([v_x, v_y, v_z])

def calculate_action_effect(self, action, ocean_current):
    """
    计算洋流对动作的影响（例如加上由阻力引起的修正）。
    """
    area = 0.3 * 0.3  # AUV横截面积
    drag_coefficient = 1.0
    relative_velocity = -ocean_current  # 假设 AUV 相对洋流静止
    drag_force = 0.5 * drag_coefficient * self.rho * area * np.linalg.norm(relative_velocity) * relative_velocity
    acceleration_due_to_drag = drag_force / self.m
    adjusted_action = np.copy(action)
    adjusted_action[:3] += acceleration_due_to_drag
    return adjusted_action

def is_valid_position(self, pos, obstacle_locs, min_distance=5.0):
    """
    检查 pos 是否与 obstacle_locs 中所有障碍物保持至少 min_distance 的距离
    """
    for obs in obstacle_locs:
        if np.linalg.norm(pos - obs) < min_distance:
            return False
    return True

def setup_start(self):
    level_config = self.config["environment"]["levels"][str(self.env_level)]
    pairs = level_config.get("start_end_pairs", [])
    if pairs and self.current_scene_index < len(pairs):
        self.start = np.array(pairs[self.current_scene_index]["start"])
    else:
        area = level_config["area"]
        depth_range = level_config["depth_range"]
        lower_bound = np.array([0, 0, -depth_range[1]])
        upper_bound = np.array([area[0]*0.2, area[1]*0.2, -depth_range[0]])
        valid = False
        while not valid:
            candidate = lower_bound + np.random.rand(3) * (upper_bound - lower_bound)
            if is_valid_position(self, candidate, self.obstacle_loc, min_distance=5.0):
                self.start = candidate
                valid = True
    print("Start point:", self.start)

def setup_end(self):
    level_config = self.config["environment"]["levels"][str(self.env_level)]
    pairs = level_config.get("start_end_pairs", [])
    if pairs and self.current_scene_index < len(pairs):
        self.end = np.array(pairs[self.current_scene_index]["end"])
    else:
        area = level_config["area"]
        depth_range = level_config["depth_range"]
        lower_bound = np.array([area[0]*0.8, area[1]*0.8, -depth_range[1]])
        upper_bound = np.array([area[0], area[1], -depth_range[0]])
        valid = False
        while not valid:
            candidate = lower_bound + np.random.rand(3) * (upper_bound - lower_bound)
            if is_valid_position(self, candidate, self.obstacle_loc, min_distance=5.0):
                self.end = candidate
                valid = True
    print("End point:", self.end)
    print(f"Distance between start and end: {np.linalg.norm(self.end - self.start):.2f}")

def get_start_end_obs(self, ind):
    self.obstacle_loc = self.predefined_obstacle_distributions[ind]
    self.obstacle_size = np.random.uniform(1, 1, 16)
    return self.start, self.end, self.obstacle_loc, self.obstacle_size

def load_obstacle_distributions(self, filename="predefined_obstacles.pkl"):
    with open(filename, 'rb') as f:
        self.predefined_obstacle_distributions = pickle.load(f)
    print(f"Predefined obstacle distributions loaded from {filename}.")

def save_obstacle_distributions(self, filename="predefined_obstacles.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(self.predefined_obstacle_distributions, f)
    print(f"Predefined obstacle distributions saved to {filename}.")

def setup_obstacles(self, num_obstacles=None, default_grid_size=(4, 5, 2), train=True):
    """
    根据当前环境配置生成障碍物分布。
    若已设置起点/终点，则以它们构成的包围盒（外扩 10%）为区域生成障碍物。
    修改后的代码支持配置文件中 obstacles 字段的不同格式：
      - 如果存在 "configurations"，则依次按照每个配置生成障碍物；
      - 如果存在 "static_count"，则使用该值生成障碍物；
      - 否则，若存在 "count_range"，则随机生成数量；
      - 默认生成 10 个障碍物。
    """
    level_config = self.config["environment"]["levels"][str(self.env_level)]
    obstacles_conf = level_config.get("obstacles", {})

    # 计算生成区域：若已设置起点和终点，则用它们构成的区域，并扩展 10%
    if hasattr(self, 'start') and hasattr(self, 'end'):
        print("AAAAA")
        lower_bound = np.minimum(self.start, self.end)
        upper_bound = np.maximum(self.start, self.end)
        margin = (upper_bound - lower_bound) * 0.1
        lower_bound = lower_bound - margin
        upper_bound = upper_bound + margin
        # 限制在关卡允许区域内
        area = level_config["area"]
        depth_range = level_config["depth_range"]
        level_lower = np.array([0, 0, -depth_range[1]])
        level_upper = np.array([area[0], area[1], depth_range[1]])
        lower_bound = np.maximum(lower_bound, level_lower)
        upper_bound = np.minimum(upper_bound, level_upper)
    else:
        print("BBBB")

        area = level_config["area"]
        depth_range = level_config["depth_range"]
        lower_bound = np.array([0, 0, -depth_range[1]])
        upper_bound = np.array([area[0], area[1], depth_range[1]])

    # 用于存放最终生成的障碍物位置
    self.obstacle_loc = []

    # ----- 分情况处理 obstacles 配置 -----
    if "configurations" in obstacles_conf:
        print("CCCCCCC")

        # 处理多个配置，每个配置独立生成障碍物
        for config in obstacles_conf["configurations"]:
            # 如果配置中提供 seed，则设置随机种子（可选）
            seed = config.get("seed", None)
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            config_count = config.get("count", 10)
            # 若配置中指定 grid_size 为 null，则采用默认值
            config_grid_size = config.get("grid_size", default_grid_size)
            if config_grid_size is None:
                config_grid_size = default_grid_size
            # 确保 grid_size 为 tuple 格式
            config_grid_size = tuple(config_grid_size)

            # 在 lower_bound 和 upper_bound 定义的区域上按照配置的 grid_size 划分网格
            x_bins = np.linspace(lower_bound[0], upper_bound[0], config_grid_size[0] + 1)
            y_bins = np.linspace(lower_bound[1], upper_bound[1], config_grid_size[1] + 1)
            z_bins = np.linspace(lower_bound[2], upper_bound[2], config_grid_size[2] + 1)
            grid_cells = [(x_bins[i], x_bins[i+1],
                           y_bins[j], y_bins[j+1],
                           z_bins[k], z_bins[k+1])
                          for i in range(config_grid_size[0])
                          for j in range(config_grid_size[1])
                          for k in range(config_grid_size[2])]
            # 如果是训练阶段，可随机选取一半的网格
            if train:
                selected_cells = random.sample(grid_cells, k=len(grid_cells)//2)
            else:
                selected_cells = grid_cells
            num_cells = len(selected_cells)
            num_in_cell = max(1, config_count // num_cells)
            for cell in selected_cells:
                x_min, x_max, y_min, y_max, z_min, z_max = cell
                obstacles_in_cell = np.random.uniform([x_min, y_min, z_min],
                                                       [x_max, y_max, z_max],
                                                       (num_in_cell, 3))
                self.obstacle_loc.extend(obstacles_in_cell)

    elif "static_count" in obstacles_conf:
        print("DDDDDDD")

        # 处理关卡 2 的情况：只生成静态障碍物
        static_count = obstacles_conf["static_count"]
        grid_size = default_grid_size
        x_bins = np.linspace(lower_bound[0], upper_bound[0], grid_size[0] + 1)
        y_bins = np.linspace(lower_bound[1], upper_bound[1], grid_size[1] + 1)
        z_bins = np.linspace(lower_bound[2], upper_bound[2], grid_size[2] + 1)
        grid_cells = [(x_bins[i], x_bins[i+1],
                       y_bins[j], y_bins[j+1],
                       z_bins[k], z_bins[k+1])
                      for i in range(grid_size[0])
                      for j in range(grid_size[1])
                      for k in range(grid_size[2])]
        if train:
            selected_cells = random.sample(grid_cells, k=len(grid_cells)//2)
        else:
            selected_cells = grid_cells
        num_cells = len(selected_cells)
        num_in_cell = max(1, static_count // num_cells)
        for cell in selected_cells:
            x_min, x_max, y_min, y_max, z_min, z_max = cell
            obstacles_in_cell = np.random.uniform([x_min, y_min, z_min],
                                                   [x_max, y_max, z_max],
                                                   (num_in_cell, 3))
            self.obstacle_loc.extend(obstacles_in_cell)

    elif "count_range" in obstacles_conf:
        print("EEEEEE")

        # 处理原有逻辑：根据 count_range 随机生成障碍物数量
        low, high = obstacles_conf["count_range"]
        num_obstacles = random.randint(low, high)
        grid_size = default_grid_size
        x_bins = np.linspace(lower_bound[0], upper_bound[0], grid_size[0] + 1)
        y_bins = np.linspace(lower_bound[1], upper_bound[1], grid_size[1] + 1)
        z_bins = np.linspace(lower_bound[2], upper_bound[2], grid_size[2] + 1)
        grid_cells = [(x_bins[i], x_bins[i+1],
                       y_bins[j], y_bins[j+1],
                       z_bins[k], z_bins[k+1])
                      for i in range(grid_size[0])
                      for j in range(grid_size[1])
                      for k in range(grid_size[2])]
        if train:
            selected_cells = random.sample(grid_cells, k=len(grid_cells)//2)
        else:
            selected_cells = grid_cells
        num_cells = len(selected_cells)
        num_in_cell = max(1, num_obstacles // num_cells)
        for cell in selected_cells:
            x_min, x_max, y_min, y_max, z_min, z_max = cell
            obstacles_in_cell = np.random.uniform([x_min, y_min, z_min],
                                                   [x_max, y_max, z_max],
                                                   (num_in_cell, 3))
            self.obstacle_loc.extend(obstacles_in_cell)

    else:
        print("FFFFFFFF")

        # 没有指定障碍物数量的情况，默认生成 10 个
        num_obstacles = 10
        grid_size = default_grid_size
        x_bins = np.linspace(lower_bound[0], upper_bound[0], grid_size[0] + 1)
        y_bins = np.linspace(lower_bound[1], upper_bound[1], grid_size[1] + 1)
        z_bins = np.linspace(lower_bound[2], upper_bound[2], grid_size[2] + 1)
        grid_cells = [(x_bins[i], x_bins[i+1],
                       y_bins[j], y_bins[j+1],
                       z_bins[k], z_bins[k+1])
                      for i in range(grid_size[0])
                      for j in range(grid_size[1])
                      for k in range(grid_size[2])]
        if train:
            selected_cells = random.sample(grid_cells, k=len(grid_cells)//2)
        else:
            selected_cells = grid_cells
        num_cells = len(selected_cells)
        num_in_cell = max(1, num_obstacles // num_cells)
        for cell in selected_cells:
            x_min, x_max, y_min, y_max, z_min, z_max = cell
            obstacles_in_cell = np.random.uniform([x_min, y_min, z_min],
                                                   [x_max, y_max, z_max],
                                                   (num_in_cell, 3))
            self.obstacle_loc.extend(obstacles_in_cell)

    self.obstacle_loc = np.array(self.obstacle_loc)

    # 确保障碍物不与起点或终点过近（距离小于 5 时重新生成）
    for i in range(len(self.obstacle_loc)):
        max_attempts = 10
        attempts = 0
        while (np.linalg.norm(self.obstacle_loc[i] - self.start) < 5 or
               np.linalg.norm(self.obstacle_loc[i] - self.end) < 5):
            if attempts >= max_attempts:
                break
            cell = random.choice(selected_cells)
            x_min, x_max, y_min, y_max, z_min, z_max = cell
            self.obstacle_loc[i] = np.random.uniform([x_min, y_min, z_min],
                                                      [x_max, y_max, z_max])
            attempts += 1

    return self.obstacle_loc

def spawn_obstacles(self, env, obstacle_locations ):
    """
    在仿真环境中生成障碍物。
    """

    prop_types = ["box", "sphere", "cylinder", "cone"]
    materials = ["white", "gold", "cobblestone", "brick", "wood", "grass", "steel", "black"]
    materials = ["wood", "grass", "steel", "black"]
    for i, location in enumerate(obstacle_locations):
        #print(f"Obstacle {i} original location: {location}")
        flipped_location = [location[0], -location[1], location[2]]
        prop_type = random.choice(prop_types)
        material = random.choice(materials)
        scale = random.uniform(0.5, 4)
        sim_physics = False
        tag = f"obstacle_{i}"
        env.spawn_prop(
            prop_type=prop_type,
            location=flipped_location,
            scale=scale,
            sim_physics=sim_physics,
            material=material,
            tag=tag
        )

def normalize_action(self, action):
    return action / self.max_action

def denormalize_action(self, normalized_action):
    return normalized_action * self.max_action