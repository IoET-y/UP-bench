import sys
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QComboBox, QCheckBox, QFormLayout, QSpinBox, QFileDialog
)


class AUVSimulatorUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AUV Simulator Configuration")
        self.setGeometry(100, 100, 600, 400)

        # 创建布局
        main_layout = QVBoxLayout()

        # 仿真环境参数
        env_layout = QFormLayout()
        self.n_targets = QSpinBox()
        self.n_targets.setRange(1, 50)
        self.n_targets.setValue(10)

        self.n_obstacles = QSpinBox()
        self.n_obstacles.setRange(0, 50)
        self.n_obstacles.setValue(20)

        self.show_viewport = QCheckBox("Show simulation Viewport")
        self.show_viewport.setChecked(True)

        self.verbose = QCheckBox("Log in Details")
        self.verbose.setChecked(False)

        env_layout.addRow("n_targets:", self.n_targets)
        env_layout.addRow("n_obstacles:", self.n_obstacles)
        env_layout.addRow(self.show_viewport)
        env_layout.addRow(self.verbose)
        main_layout.addLayout(env_layout)

        # 规划器选择
        self.route_selector = QComboBox()
        self.route_selector.addItems(["rrt", "astar", "djs", "GA", "FA", "ACO", "SAC_LQR", "sac"])
        self.route_selector.currentTextChanged.connect(self.update_planner_params)
        main_layout.addWidget(QLabel("algorithm selector:"))
        main_layout.addWidget(self.route_selector)

        # 规划器参数布局
        self.planner_params_layout = QFormLayout()
        self.param_inputs = {}
        main_layout.addLayout(self.planner_params_layout)

        # 训练参数
        train_layout = QFormLayout()
        self.num_episodes = QSpinBox()
        self.num_episodes.setRange(1, 100000)
        self.num_episodes.setValue(5000)

        self.max_step = QSpinBox()
        self.max_step.setRange(1, 50000)
        self.max_step.setValue(10000)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 2048)
        self.batch_size.setValue(32)

        train_layout.addRow("num_episodes:", self.num_episodes)
        train_layout.addRow("max_step:", self.max_step)
        train_layout.addRow("batch_size:", self.batch_size)
        main_layout.addLayout(train_layout)

        # 选择模型路径
        self.model_path_input = QLineEdit()
        self.model_path_input.setText("as.pth")
        self.select_model_btn = QPushButton("model path")
        self.select_model_btn.clicked.connect(self.select_model_path)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("model path:"))
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(self.select_model_btn)
        main_layout.addLayout(model_layout)

        # 运行按钮
        self.run_button = QPushButton("Running simulation")
        self.run_button.clicked.connect(self.run_simulation)
        main_layout.addWidget(self.run_button)

        self.setLayout(main_layout)
        self.update_planner_params()

    def update_planner_params(self):
        """ 更新不同路径规划算法的参数输入框 """
        # 清空已有参数
        for i in reversed(range(self.planner_params_layout.count())):
            self.planner_params_layout.itemAt(i).widget().deleteLater()
        self.param_inputs.clear()

        route = self.route_selector.currentText()

        planner_params = {
            "ACO": ["num_ants", "iterations", "alpha", "beta", "evaporation_rate", "Q"],
            "SAC_LQR": ["sensor_range"],
            "sac": []
        }

        if route in planner_params:
            for param in planner_params[route]:
                input_field = QLineEdit()
                self.planner_params_layout.addRow(f"{param}:", input_field)
                self.param_inputs[param] = input_field

    def select_model_path(self):
        """ 选择模型文件路径 """
        file_path, _ = QFileDialog.getOpenFileName(self, "select model file", "", "(*.pth *.pt)")
        if file_path:
            self.model_path_input.setText(file_path)

    def run_simulation(self):
        """ 运行仿真代码 """
        cmd = [
            sys.executable, "run_single_algorithm.py",
            "--route", self.route_selector.currentText(),
            "--num_episodes", str(self.num_episodes.value()),
            "--max_step", str(self.max_step.value()),
            "--batch_size", str(self.batch_size.value()),
            "--model_path", self.model_path_input.text()
        ]

        # 添加仿真环境参数
        cmd.extend(["--n_targets", str(self.n_targets.value())])
        cmd.extend(["--n_obstacles", str(self.n_obstacles.value())])

        if self.show_viewport.isChecked():
            cmd.append("--show")
        if self.verbose.isChecked():
            cmd.append("--verbose")

        # 添加特定 planner 的参数
        for param, widget in self.param_inputs.items():
            cmd.extend([f"--{param}", widget.text()])

        print("Running:", " ".join(cmd))
        subprocess.run(cmd)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AUVSimulatorUI()
    window.show()
    sys.exit(app.exec())