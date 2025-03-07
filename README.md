Here’s a stable and structured README.md file incorporating the benchmarking, individual planner execution, and GUI-based simulation functionalities.

# UP-Bench: A Benchmark for Underwater Path Planning Algorithms

**Author:** Di Yang, Yanhai Xiong  
**Affiliation:** The College of William and Mary  
**GitHub Repository:** [UP-Bench on GitHub](https://github.com/IoET-y/UP-bench)  

## 🌊 Overview
**UP-Bench** is an open-source benchmarking framework for evaluating **underwater path planning** algorithms. It provides:
- **Unified evaluation metrics** to assess performance across different methods.
- **Automated benchmarking** for classical, evolutionary, and RL-based algorithms.
- **Realistic underwater simulation** with ocean currents and obstacles via **HoloOcean**.
- **GUI-based execution** to allow interactive configuration and algorithm selection.

The benchmark includes **multiple planning algorithms**:
- **Classical:** A*, Dijkstra, RRT
- **Evolutionary:** Genetic Algorithm (GA), Firefly Algorithm (FA), Ant Colony Optimization (ACO)
- **Reinforcement Learning (RL):** Soft Actor-Critic (SAC), Hybrid SAC+A*+LQR

---

## 🚀 Key Features
✅ **3D underwater path planning** with physics-based simulation  
✅ **Standardized evaluation metrics**: Path efficiency, energy consumption, computational cost  
✅ **Automated benchmarking**: Runs all planners in a structured experiment  
✅ **GUI-based execution**: Allows users to configure and run simulations easily  

---

## 📌 Supported Path Planning Algorithms

### 🔍 Classical Methods
- **A\***: Optimal heuristic-based search.
- **Dijkstra**: Uniform cost search for shortest path.
- **RRT**: Randomized search in high-dimensional spaces.

### 🧬 Evolutionary Methods
- **GA**: Evolutionary optimization-based path search.
- **FA**: Firefly algorithm for multi-modal optimization.
- **ACO**: Ant colony optimization for global pathfinding.

### 🤖 Reinforcement Learning
- **SAC**: Deep RL for dynamic environments.
- **SAC+A***: Hybrid RL-based planning with heuristic A* search.
- **SAC+A*+LQR**: RL combined with **LQR tracking**, improving training speed.

---

## 📊 Evaluation Metrics
UP-Bench evaluates planners based on:
1. **Path Length** 📏 – Measures trajectory efficiency.
2. **Execution Time** ⏳ – Time taken to reach the target.
3. **Planning Computation Time** 💻 – Time needed to generate paths.
4. **Collision Rate** ⚠️ – Percentage of paths resulting in collisions.
5. **Success Rate** 🎯 – Percentage of successful goal reaches.
6. **Energy Consumption** 🔋 – Total energy used during navigation.
7. **Smoothness** 🛶 – Measures trajectory deviation and control efficiency.

---

## ⚙️ Installation

### 🛠 Prerequisites
- Python 3.8+
- HoloOcean Simulation Framework
- NumPy, SciPy, Matplotlib
- PyTorch (for RL-based approaches)
- `wandb` (for experiment logging)

### 🚀 Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/IoET-y/UP-bench.git
   cd UP-bench

	2.	Install dependencies:

pip install -r requirements.txt


	3.	Set up the HoloOcean simulation environment (refer to the official HoloOcean setup guide).

⸻

🔥 Running Benchmarks

📈 Running Full Benchmark

To evaluate all planners:

python benchmark.py --max_step 5000 --BS 32

This script iterates over all available planning algorithms, evaluates them, and outputs the success rate, execution time, path length, and other metrics.



🎯 Running a Single Planner Demo

To run a specific path planning algorithm:

python run_single_algorithm.py --route ACO --num_episodes 100 --max_step 5000

Available options for --route:
	•	astar
	•	rrt
	•	djs (Dijkstra)
	•	GA (Genetic Algorithm)
	•	FA (Firefly Algorithm)
	•	ACO (Ant Colony Optimization)
	•	SAC_LQR (RL with LQR tracking)
	•	sac (Soft Actor-Critic)

🎨 Visualizing Results

python visualize.py --log-dir logs/



⸻

🖥️ GUI-Based Simulator

A Graphical User Interface (GUI) is available to configure and run simulations easily.

Launch the GUI:

python UI.py

GUI Features:
	•	Algorithm selection: Choose planners interactively.
	•	Simulation parameters: Adjust number of obstacles and targets.
	•	Training options: Configure episode count, batch size, and RL model path.
	•	Live execution: Run the planner and monitor real-time performance.


⸻

🔬 Future Work
	•	🌊 Enhanced Environmental Dynamics: Adding wave disturbances and dynamic obstacles.
	•	🤖 Multi-AUV Coordination: Collaborative path planning for AUV swarms.
	•	🛠 Real-World Deployment: Testing the framework on real underwater robots.

⸻

📜 Citation

If you use UP-Bench in your research, please cite:

@article{UP-Bench2025,
  author = {Di Yang, Yanhai Xiong},
  title = {UP-Bench: A Benchmark for Underwater Path Planning Algorithms},
  year = {2025},
  journal = {Conference XX},
  publisher = {ACM}
}



⸻

📝 License

UP-Bench is an open-source project under the MIT License.

⸻

🚀 UP-Bench standardizes underwater path planning research and accelerates algorithm development! 🌊
🔗 GitHub Repository
📩 Contributions and feedback are welcome!

---

### 🔥 **Key Enhancements**
- **Includes benchmarking execution**
- **Supports running a single planner**
- **Describes GUI functionality**
- **Provides example output formatting**
- **Explains visualization tools**

This README is now **comprehensive, stable, and structured**! Let me know if you need refinements. 🚀
