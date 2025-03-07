import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import holoocean
from tqdm import tqdm
from auv_planning.planning.ASTAR_2025 import AStarPlanner
from auv_planning.planning.RRT_2025 import RRTPlanner
from auv_planning.planning.DJS_2025 import DijkstraPlanner
from auv_planning.planning.GA_2025 import GeneticPlanner
from auv_planning.planning.ACO_2025 import ACOPlanner
from auv_planning.planning.FA_2025 import FireflyPlanner

from auv_planning.planning.RTASTART_2025 import RTAAStarPlanner
from auv_planning.planning.RRTOL_2025 import OnlineRRTStarPlanner
from auv_planning.planning.PFM_2025 import PFMPlanner
from auv_planning.planning.Theta_2025 import RSAPPlanner
from auv_planning.planning.SAC_2025 import SACPlanner
from auv_planning.planning.SAC_LQR_20251 import SACLQRPlanner

from auv_planning.holoocean_config import scenario
from auv_planning.custom_env import custom_environment

sns.set(context="paper", style="whitegrid", font_scale=0.8)


def main(show, plot, verbose, route, num_episodes, max_step, model_path, BS):
#check if need to install holoocean
    if "Ocean" not in holoocean.installed_packages():
        holoocean.install("Ocean")

    # initialize holoocean
    print("Starting HoloOcean environment initialization...")
    dummy_env = custom_environment(scenario_cfg=scenario, n_targets=10, n_obstacles=20, show_viewport=show, verbose=verbose)
    print("Environment initialized successfully.")

    # initialize planner
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
    elif route == "PFM":
        planner = PFMPlanner(grid_resolution=0.5, max_steps=max_step)
    elif route == "RTTOL":
        planner = OnlineRRTStarPlanner(grid_resolution=0.5, max_steps=max_step)
    elif route == "RTAST":
        planner = RTAAStarPlanner(grid_resolution=0.5, max_steps=max_step)
    elif route == "RSAP":
        planner = RSAPPlanner(grid_resolution=0.5, max_steps=max_step)
    if planner is not None:
        planner.train(dummy_env)
        print(f"Using {route} planner.")



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
    parser.add_argument("--route", type=str, default="RSAP", help="select planner")
    parser.add_argument("--num_episodes", type=int, default=5000, help="total runing episode")
    parser.add_argument("--max_step", type=int, default=10000, help="maxstep per episode")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for reinforcement learning")
    parser.add_argument("--model_path", type=str, default="as.pth", help="RL model storage path")
    parser.add_argument("--show", action="store_true", help="show simulation windows")
    parser.add_argument("--plot", action="store_true", help="plot or noit")
    parser.add_argument("--verbose", action="store_true", help="Log or not")
    parser.add_argument("--n_targets", type=int, default=10, help="Targets number")
    parser.add_argument("--n_obstacles", type=int, default=20, help="Obstacles number")

    args = parser.parse_args()

    main(show=args.show, plot=args.plot, verbose=args.verbose, route=args.route,
         num_episodes=args.num_episodes, max_step=args.max_step,
         model_path=args.model_path, BS=args.batch_size)
