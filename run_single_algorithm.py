import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import holoocean
from tqdm import tqdm
from auv_planning.planning.SAC_2025 import SACPlanner
from auv_planning.planning.ASTAR_2025 import AStarPlanner
from auv_planning.planning.RRT_2025 import RRTPlanner
from auv_planning.planning.DJS_2025 import DijkstraPlanner
from auv_planning.planning.GA_2025 import GeneticPlanner
from auv_planning.planning.ACO_2025 import ACOPlanner
from auv_planning.planning.FA_2025 import FireflyPlanner
from auv_planning.planning.SAC_LQR_20251 import SACLQRPlanner
from auv_planning.planning import Traj
from plotter import Plotter
from auv_planning.holoocean_config import scenario
from auv_planning.custom_env import custom_environment

sns.set(context="paper", style="whitegrid", font_scale=0.8)


def main(show, plot, verbose, route, num_episodes, max_step, model_path, BS):
    # 检查 holoocean 环境是否安装
    if "Ocean" not in holoocean.installed_packages():
        holoocean.install("Ocean")

    # 初始化仿真环境
    print("Starting HoloOcean environment initialization...")
    dummy_env = custom_environment(scenario_cfg=scenario, n_targets=10, n_obstacles=20, show_viewport=show, verbose=verbose)
    print("Environment initialized successfully.")

    # 初始化控制器和规划器
    planner = None
    if route == "rrt":
        planner = RRTPlanner(grid_resolution=0.5, max_steps=max_step)
    elif route == "astar":
        planner = AStarPlanner(grid_resolution=0.5, max_steps=max_step)
    elif route == "djs":
        planner = DijkstraPlanner(grid_resolution=0.5, max_steps=max_step)
    elif route == "GA":
        planner = GeneticPlanner(grid_resolution=0.5, max_steps=max_step)
    elif route == "FA":
        planner = FireflyPlanner(grid_resolution=0.5, max_steps=max_step)
    elif route == "ACO":
        planner = ACOPlanner(grid_resolution=0.5, max_steps=max_step, num_ants=50, iterations=100,
                             alpha=1.0, beta=4.0, evaporation_rate=0.1, Q=100.0,
                             max_path_steps=max_step)
    elif route == "SAC_LQR":
        planner = SACLQRPlanner(sensor_range=10.0, grid_resolution=0.5, max_steps=max_step, batch_size=BS)
        planner.train(dummy_env, model_path=model_path)
    elif route == "sac":
        planner = SACPlanner(batch_size=BS, config_file="config_all.yaml")
        planner.train(dummy_env, model_path=model_path)
    else:
        planner = Traj(route, num_episodes)

    if planner is not None:
        planner.train(dummy_env)
        print(f"Using {route} planner.")

    # 如果需要绘图
    if plot:
        plotter = Plotter(["True", "Estimated", "Desired"])


if __name__ == "__main__":
    def str2bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in ('true', '1', 'yes'):
            return True
        elif value.lower() in ('false', '0', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")


    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=str, default="ACO", help="选择使用的规划算法")
    parser.add_argument("--num_episodes", type=int, default=5000, help="训练的总回合数")
    parser.add_argument("--max_step", type=int, default=10000, help="每回合最大步数")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--model_path", type=str, default="as.pth", help="模型存储路径")
    parser.add_argument("--show", action="store_true", help="是否显示仿真窗口")
    parser.add_argument("--plot", action="store_true", help="是否显示绘图")
    parser.add_argument("--verbose", action="store_true", help="是否启用详细日志")
    parser.add_argument("--n_targets", type=int, default=10, help="目标物数量")
    parser.add_argument("--n_obstacles", type=int, default=20, help="障碍物数量")

    args = parser.parse_args()

    main(show=args.show, plot=args.plot, verbose=args.verbose, route=args.route,
         num_episodes=args.num_episodes, max_step=args.max_step,
         model_path=args.model_path, BS=args.batch_size)