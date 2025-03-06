# parallel_train.py
import holoocean
import wandb
from multiprocessing import Pool
#from auv_planning.planning.LQR_NPDDQN import LQRNPDDQNPlanner
import torch.multiprocessing as mp

def run_single_env(planner, num_episodes, env_config, verbose):
    """每个进程运行单个环境的训练"""

    # 初始化 WandB
    wandb.init(project="auv_control_project", name=f"NPDDQN_training_worker_{mp.current_process().name}")

    env = holoocean.make(scenario_cfg=env_config, show_viewport=False, verbose=verbose)
    planner.train(env, num_episodes=num_episodes)
    
    # WandB 结束
    wandb.finish()

    return planner.memory

def parallel_train(planner, num_episodes, num_workers, verbose, scenario):
    """并行训练多个环境"""
    # 共享网络参数
    planner.policy_net.share_memory()

    # 创建多进程池
    with Pool(num_workers) as pool:
        results = [
            pool.apply_async(run_single_env, (planner, num_episodes, scenario, verbose))
            for _ in range(num_workers)
        ]
        pool.close()
        pool.join()

    # 汇总每个进程的经验池
    for result in results:
        worker_memory = result.get()
        planner.memory.extend(worker_memory)

    # 保存训练好的模型
    planner.save_model(episode=num_episodes * num_workers, path='npddqn_parallel_model.pth')
    print("并行训练完成，模型已保存。")