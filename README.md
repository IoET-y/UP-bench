Here’s the README in Markdown format:

# UP-Bench: A Benchmark for Underwater Path Planning Algorithms

**Author:** Di Yang, Yanhai Xiong  
**Affiliation:** The College of William and Mary  
**GitHub Repository:** [UP-Bench on GitHub](https://github.com/IoET-y/UP-bench)  

## Overview
**UP-Bench** is an open-source benchmarking platform for **underwater path planning** algorithms, providing a **unified evaluation framework**, **automated performance assessment**, and **reproducible experimental workflows**. Built on the **HoloOcean** simulation platform, UP-Bench incorporates realistic underwater dynamics, including:
- 🌊 **Ocean currents**
- 🏗️ **Static and dynamic obstacles**
- 📡 **Sensor models**

The benchmark supports a wide range of **path planning algorithms**, including:
- **Classical algorithms**: A*, Dijkstra, RRT
- **Evolutionary methods**: Genetic Algorithm (GA), Ant Colony Optimization (ACO)
- **Reinforcement Learning (RL)**: Soft Actor-Critic (SAC), Hybrid SAC+A*+LQR

## 🔥 Key Features
✅ **3D underwater path planning** with realistic dynamics and ocean currents  
✅ **Standardized evaluation metrics**: Path efficiency, success rate, collision rate, energy consumption, computational cost  
✅ **Automated benchmarking tools** for systematic algorithm comparisons  
✅ **Modular API** allowing easy integration of new algorithms  

---

## 📌 Supported Path Planning Algorithms

### 🏛️ Classical Methods
- **A\***: Guarantees optimality with heuristic search.
- **Dijkstra**: Finds the shortest path using uniform cost search.
- **RRT**: Efficient for high-dimensional search spaces.

### 🧬 Evolutionary Methods
- **GA**: Optimizes paths considering multiple objectives.
- **ACO**: Uses pheromone-based reinforcement for path optimization.

### 🤖 Reinforcement Learning
- **SAC**: Learns policies for dynamic underwater environments.
- **SAC+A***: Uses SAC for sub-goal selection and A* for efficient path generation.
- **SAC+A*+LQR**: Hybrid RL method with **LQR tracking**, reducing training time significantly.

---

## 📊 Evaluation Metrics
UP-Bench evaluates path planning algorithms based on:
1. **Path Length** 📏 – Measures trajectory efficiency.
2. **Execution Time** ⏳ – Time required for the AUV to complete navigation.
3. **Planning Computation Time** 💻 – Time needed to compute/re-plan paths.
4. **Collision Rate** ⚠️ – Percentage of paths resulting in collisions.
5. **Success Rate** 🎯 – Percentage of successful goal reaches.
6. **Energy Consumption** 🔋 – Total energy expended during navigation.
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

📌 Usage

🔍 Running a Path Planning Algorithm

To execute a specific path planning algorithm:

python run.py --algorithm ACO --env OpenWater

Available options for --algorithm: A*, Dijkstra, RRT, GA, ACO, SAC, SAC+A*, SAC+A*+LQR.

🤖 Training an RL-based Planner

For training an RL-based path planner:

python train.py --algorithm SAC --episodes 1000

📈 Benchmarking & Evaluation

Run benchmarking experiments:

python benchmark.py --num-runs 50 --env OpenWater

🎨 Visualizing Results

Use the built-in visualization tools:

python visualize.py --log-dir logs/



⸻

🚀 Future Work
	•	🌊 Advanced Environmental Dynamics: Introducing variable water densities, wave disturbances, and temperature gradients.
	•	🤖 Multi-AUV Coordination: Swarm-based path planning.
	•	🛠 Real-World Validation: Field trials with physical AUVs.
	•	📊 Expanded Sensor Modalities: Incorporating high-fidelity sonar imaging, acoustic localization, and sensor fusion.

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

UP-Bench is an open-source project. Refer to the LICENSE file for details.

⸻

🚀 UP-Bench aims to standardize algorithmic comparisons and accelerate research in underwater path planning! 🌊
🔗 GitHub Repository
📩 Contributions and feedback are welcome!

This markdown-formatted README provides a **structured, professional, and visually appealing** overview of UP-Bench. Let me know if you need modifications! 🚀
