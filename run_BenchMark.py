import sys
import os
import numpy as np
import seaborn as sns
import holoocean
from tqdm import tqdm

from auv_planning.planning.ASTAR_2025 import AStarPlanner
from auv_planning.planning.DJS_2025 import DijkstraPlanner
from auv_planning.planning.RRT_2025 import RRTPlanner
from auv_planning.planning.GA_2025 import GeneticPlanner
from auv_planning.planning.FA_2025 import FireflyPlanner
from auv_planning.planning.ACO_2025 import ACOPlanner

from auv_planning.planning.RTASTART_2025 import RTAAStarPlanner
from auv_planning.planning.RRTOL_2025 import OnlineRRTStarPlanner
from auv_planning.planning.PFM_2025 import PFMPlanner
from auv_planning.planning.Theta_2025 import RSAPPlanner
from auv_planning.planning.SAC_2025 import SACPlanner
from auv_planning.planning.SAC_LQR_20251 import SACLQRPlanner

from auv_planning.holoocean_config import scenario
from auv_planning.custom_env import custom_environment

sns.set(context="paper", style="whitegrid", font_scale=0.8)

from mpl_toolkits.mplot3d import Axes3D


def benchmark(show=True, verbose=False, max_step=1024, BS=32):
    """
    依次对所有规划算法进行训练，并收集评估指标：
      返回的评估指标为一个元组： (successrate, ave_path_length, ave_excu_time, ave_plan_time, ave_smoothness, ave_energy)
    """
    # 检查 holoocean 环境中是否安装了 Ocean 包
    if "Ocean" not in holoocean.installed_packages():
        holoocean.install("Ocean")

    # 根据 scenario 配置计算仿真时间间隔
    ts = 1 / scenario["ticks_per_sec"]

    print("Starting HoloOcean environment initialization for benchmark...")

    # 用于存放每个算法评估指标的字典
    results = {}

    # 定义各个规划器的构造函数（部分规划器需要特殊参数）
    planners = {
        "AStar": lambda: AStarPlanner( grid_resolution=0.5, max_steps=max_step),
        "Dijkstra": lambda: DijkstraPlanner( grid_resolution=0.5, max_steps=max_step),
        "RRT": lambda: RRTPlanner( grid_resolution=0.5, max_steps=max_step),
        "GA": lambda: GeneticPlanner( grid_resolution=0.5, max_steps=max_step),
        "FA": lambda: FireflyPlanner( grid_resolution=0.5, max_steps=max_step),
        "ACO": lambda: ACOPlanner( grid_resolution=0.5, max_steps=max_step,
                                  num_ants=100, iterations=100, alpha=1.0, beta=3.0,
                                  evaporation_rate=0.1, Q=100.0, max_path_steps=max_step),
        "RTAAStar": lambda: RTAAStarPlanner( grid_resolution=0.5, max_steps=max_step),
        "OnlineRRTStar": lambda: OnlineRRTStarPlanner( grid_resolution=0.5, max_steps=max_step),
        "PFM": lambda: PFMPlanner( grid_resolution=0.5, max_steps=max_step),
        "RSAP": lambda: RSAPPlanner( grid_resolution=0.5, max_steps=max_step),
        "SAC": lambda: SACPlanner( batch_size=BS, config_file="config_all.yaml"),
        "SAC_LQR": lambda: SACLQRPlanner( sensor_range=10.0, grid_resolution=0.5, max_steps=max_step, batch_size=BS)
    }

    # 对于每个规划器，分别创建新的环境实例并执行 train 训练
    for name, planner_constructor in planners.items():
        print(f"\nRunning benchmark for {name} planner...")
        # 为每个算法创建一个新的仿真环境实例
        env = custom_environment(scenario_cfg=scenario, n_targets=10, n_obstacles=40, show_viewport=show, verbose=verbose)
        planner = planner_constructor()
        if name == "SAC_LQR":
            eval_metrics = planner.train(env, model_path=model_path)
        elif name == "SAC":
            eval_metrics = planner.train(env,num_episodes=num_episodes, max_steps=max_step, model_path=model_path)
        else:
            eval_metrics = planner.train(env)
        results[name] = eval_metrics
        print(f"{name} evaluation metrics: {eval_metrics}")

    # 汇总输出所有算法的评估指标
    print("\nBenchmark Results Summary:")
    header = ["Planner", "Success Rate", "Avg Path Length", "Avg Excu Time", "Avg Plan Time", "Avg Smoothness", "Avg Energy"]
    print("{:<15} {:<15} {:<18} {:<18} {:<15} {:<15} {:<15}".format(*header))
    for name, metrics in results.items():
        # 假定返回的 metrics 为一个包含六个指标的元组
        print("{:<15} {:<15} {:<18} {:<18} {:<15} {:<15} {:<15}".format(
            name,
            str(metrics[0]),
            str(metrics[1]),
            str(metrics[2]),
            str(metrics[3]),
            str(metrics[4]),
            str(metrics[5])
        ))
    return results


if __name__ == "__main__":
    benchmark_results = benchmark(show=True, verbose=False, max_step=5000, BS=32)
    print("\nBenchmark completed.\n")