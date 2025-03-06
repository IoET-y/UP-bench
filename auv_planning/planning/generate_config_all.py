# generate_config_all.py
import yaml

def generate_config_all():
    config = {
        "seed": 42,
        "environment": {
            # 默认选用的环境级别（可在运行时修改此值）
            "level": "1",
            "levels": {
                "1": {
                    "name": "Level 1: 小尺寸、浅水、弱洋流、少量固定障碍",
                    "area": [100, 100],           # 长×宽（单位：米）
                    "depth_range": [0, 30],         # 水深范围：0~30m
                    "obstacles": {
                        "configurations": [
                            {"seed": 1001, "count": 7, "grid_size": None},
                            {"seed": 1002, "count": 8, "grid_size": [4, 5, 1]}
                        ]
                    },
                    "start_end_pairs": [
                        {"start": [10, 10, -5], "end": [90, 90, -5]},
                        {"start": [5, 5, -5], "end": [95, 95, -5]},
                        {"start": [15, 15, -5], "end": [85, 85, -5]},
                        {"start": [10, 10, -5], "end": [90, 90, -15]},
                        {"start": [5, 5, -5], "end": [95, 95, -15]},
                        {"start": [15, 15, -5], "end": [85, 85, -15]}

                    ],
                    "current": {
                        "speed": 0.2,
                        "direction": [1, 0, 0]
                    }
                },
                "2": {
                    "name": "Level 2: 中等尺寸、适度洋流、少量动态障碍",
                    "area": [300, 300],
                    "depth_range": [0, 80],
                    "obstacles": {
                        "static_count": 15,
                        "dynamic_count_range": [2, 5]
                    },
                    "start_end_pairs": [
                        {"start": [20, 20, -10], "end": [280, 280, -10]},
                        {"start": [30, 30, -15], "end": [270, 270, -15]},
                        {"start": [25, 25, -12], "end": [275, 275, -12]}
                    ],
                    "current": {
                        "speed": 0.5,
                        "variation": "zone"  # 表示洋流在不同区域略有变化
                    }
                },
                "3": {
                    "name": "Level 3: 大型深水场景、强洋流、多样障碍（含动态）",
                    "area": [1000, 1000],
                    "depth_range": [0, 200],
                    "obstacles": {
                        "configurations": [
                            {"seed": 3001, "count": 40, "grid_size": [6, 6, 3]}
                        ]
                    },
                    "start_end_pairs": [
                        {"start": [50, 50, -20], "end": [950, 950, -50]},
                        {"start": [100, 100, -30], "end": [900, 900, -40]}
                    ],
                    "current": {
                        "speed_range": [0.8, 1.5],
                        "disturbance": True  # 表示中途存在随机扰动
                    }
                },
                "X": {
                    "name": "Level X: 自定义极端场景",
                    "custom": True,
                    "area": [500, 500],
                    "depth_range": [0, 100],
                    "obstacles": {
                        "configurations": [
                            {"seed": 4001, "count": 50, "grid_size": [5, 5, 2]}
                        ]
                    },
                    "start_end_pairs": [
                        {"start": [50, 50, -10], "end": [450, 450, -10]}
                    ],
                    "current": {
                        "speed": 2.0,
                        "direction": [0, 1, 0],
                        "disturbance": True
                    }
                }
            }
        },
        "metrics": {
            "path_length": True,
            "time_to_completion": True,
            "collision_count": True,
            "success_rate": True,
            "energy_consumption": True,
            "smoothness": True,
            "planning_time": True,
            "robustness": True,
            "multi_goal_completion": True
        }
    }
    return config

if __name__ == "__main__":
    config = generate_config_all()
    with open("config_all.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    print("配置文件 config_all.yaml 生成完毕！")