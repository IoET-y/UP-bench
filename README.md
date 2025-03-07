Based on the content of your paper UP-Bench: A Benchmark for Underwater Path Planning Algorithms, here is a README description for your code repository:

⸻

UP-Bench: A Benchmark for Underwater Path Planning Algorithms

Author: Di Yang, Yanhai Xiong
Affiliation: The College of William and Mary
GitHub Repository: https://github.com/IoET-y/UP-bench

Overview

UP-Bench is an open-source benchmarking platform for underwater path planning algorithms, providing a unified evaluation framework, automated performance assessment, and reproducible experimental workflows. Built on the HoloOcean simulation platform, UP-Bench incorporates realistic underwater dynamics, including:
	•	Ocean currents
	•	Static and dynamic obstacles
	•	Sensor models

The benchmark supports a wide range of path planning algorithms, including:
	•	Classical algorithms: A*, Dijkstra, RRT
	•	Evolutionary methods: Genetic Algorithm (GA), Ant Colony Optimization (ACO)
	•	Reinforcement Learning (RL): Soft Actor-Critic (SAC), Hybrid SAC+A*+LQR

Key Features
	•	3D underwater path planning with realistic underwater dynamics and self-edited ocean currents.
	•	Standardized evaluation metrics: Path efficiency, success rate, collision rate, energy consumption, computational cost.
	•	Automated benchmarking tools for systematic algorithm comparisons.
	•	Modular API that allows easy integration of new algorithms.

Supported Path Planning Algorithms

Classical Methods:
	•	A*: Guarantees optimality with a heuristic search.
	•	Dijkstra: Finds the shortest path using uniform cost search.
	•	RRT: Efficient for high-dimensional search spaces.

Evolutionary Methods:
	•	GA: Optimizes paths considering multiple objectives.
	•	ACO: Uses pheromone-based path reinforcement.

Reinforcement Learning:
	•	SAC: Trains policies for dynamic underwater environments.
	•	SAC+A*: Uses SAC for sub-goal selection and A* for efficient path generation.
	•	SAC+A+LQR*: Hierarchical reinforcement learning with LQR tracking, reducing training time significantly.

Evaluation Metrics

UP-Bench evaluates path planning algorithms based on:
	1.	Path Length: Efficiency of the generated trajectory.
	2.	Execution Time: Time taken to complete navigation.
	3.	Planning Computation Time: Time required to compute/re-plan the path.
	4.	Collision Rate: Percentage of paths that result in collisions.
	5.	Success Rate: Percentage of successful goal reaches.
	6.	Energy Consumption: Total energy expended during navigation.
	7.	Smoothness: Sudden trajectory deviations and control efficiency.

Installation

Prerequisites
	•	Python 3.8+
	•	HoloOcean Simulation Framework
	•	NumPy, SciPy, Matplotlib
	•	PyTorch (for RL-based approaches)
	•	wandb (for experiment logging)

Installation Steps
	1.	Clone the repository:

git clone https://github.com/IoET-y/UP-bench.git
cd UP-bench


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Set up the HoloOcean simulation environment (refer to the official HoloOcean setup guide).

Usage

Running a Path Planning Algorithm

To execute a specific path planning algorithm, run:

python run.py --algorithm ACO --env OpenWater

Available options for --algorithm: A*, Dijkstra, RRT, GA, ACO, SAC, SAC+A*, SAC+A*+LQR.

Training an RL-based Planner

For training an RL-based path planner, use:

python train.py --algorithm SAC --episodes 1000

Benchmarking & Evaluation

To run benchmarking experiments:

python benchmark.py --num-runs 50 --env OpenWater

Visualizing Results

Use the built-in visualization tools:

python visualize.py --log-dir logs/

Future Work
	•	Advanced Environmental Dynamics: Integration of time-varying ocean currents and turbulence.
	•	Multi-AUV Coordination: Swarm-based path planning.
	•	Real-World Validation: Field trials with physical AUVs.

Citation

If you use UP-Bench in your research, please cite:

@article{UP-Bench2025,
  author = {Di Yang, Yanhai Xiong},
  title = {UP-Bench: A Benchmark for Underwater Path Planning Algorithms},
  year = {2025},
  journal = {Conference XX},
  publisher = {ACM}
}



⸻

This README provides a comprehensive overview of the UP-Bench project and its functionality. Let me know if you need further refinements! 🚀
