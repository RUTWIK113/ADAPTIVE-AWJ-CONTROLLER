# Adaptive Control System for Abrasive Waterjet (AWJ) Cutting

![Language](https://img.shields.io/badge/Language-Python-blue)

A closed-loop adaptive control system for the Abrasive Waterjet (AWJ) cutting process. This project uses a **neuro-genetic algorithm** to optimize process parameters in real-time, compensating for machine wear (like nozzle diameter changes) to maintain a consistent, desired depth of cut.

This system is based on the research paper: "An adaptive control strategy for the abrasive waterjet cutting process..."

---
## 🎯 Core Features

* **ANN Process Model:** An Artificial Neural Network (ANN) is trained on experimental or physics-based data to act as a fast, accurate predictor of the AWJ process. It learns the complex relationship between inputs and the resulting depth of cut.
* **Genetic Algorithm Optimizer:** A Genetic Algorithm (GA) uses the trained ANN to perform a smart search for the optimal set of controllable parameters (like pressure, abrasive flow, and traverse speed) that will achieve a target depth of cut.
* **Machine Vision (Concept):** Includes the algorithmic foundation for a machine vision module (`monitoring.py`) to measure nozzle wear using Canny edge detection and Hough transforms.
* **Scalable & Adaptable:** The system is designed to be trained on any dataset (real or synthetic) and can be configured to use different inputs (e.g., focusing nozzle diameter vs. orifice diameter).

---
## 📁 Project Structure
Adaptive_Control_of_AWJ/
├── .git/
├── .idea/
├── .venv/
├── control/
│   ├── __init__.py
│   ├── ann_model.py
│   └── ga_optimizer.py
├── data/
│   ├── .gitkeep
│   ├── awj_model.keras
│   ├── awj_physics_training_data.csv
│   └── scaler.pkl
├── vision/
│   ├── __init__.py
│   ├── monitoring.py
│   └── test_images/
│       └── nozzle_tip.jpg
├── .gitignore
├── create_dataset_physics.py
├── create_dummy_model.py
├── main.py
├── README.md
└── train_model.py

---
## 🚀 How to Use

Follow these steps to get the system running.

### Prerequisites

* Python 3.8+
* Git
* Required Python libraries:
    ```bash
    pip install tensorflow pandas numpy scikit-learn deap opencv-python-headless
    ```

### Step 1: Add Your Data

1.  Place your dataset (e.g., `author_data.csv`) inside the `data/` folder.
2.  Open `train_model.py`.
3.  Change the `DATA_FILE` variable to point to your new file:
    ```python
    DATA_FILE = os.path.join('data', 'author_data.csv')
    ```
4.  Change the `FOURTH_INPUT_COLUMN` variable to match the name of the diameter column in your CSV (e.g., `"focusing_nozzle_diameter"`).

### Step 2: Train the Model

Run the training script from your terminal. This will read your CSV, train the ANN, and save the `awj_model.keras` and `scaler.pkl` files.

```bash
python train_model.py
