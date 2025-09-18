# Liquid State Machine for Trajectory Classification (32×32)

[![Example Repo](https://img.shields.io/badge/example-repo-blue)](https://github.com/RuggeroFreddi/Example-SNNpy)
[![PyPI](https://img.shields.io/pypi/v/snn-reservoir-py.svg)](https://pypi.org/project/snn-reservoir-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/snn-reservoir-py.svg)](https://pypi.org/project/snn-reservoir-py/)

This repository demonstrates how to use the `SNN` class from **snnpy** ([**snnpy** – GitHub repository](https://github.com/RuggeroFreddi/snnpy)) (`from snnpy.snn import SNN, SimulationParams, STDPParams`, installable via `pip install snn-reservoir-py==2.0.0`). The experiment implements a **Liquid State Machine (LSM)** that classifies the **trajectory of a ball in a video** among several possible classes, despite **noise** and **jitter**.
<p align="center">
  <img src="video/seven_gestures.gif" alt="Seven gestures: example of generated trajectories" width="480" />
</p>

## ✅ What the experiment shows

With an **average accuracy above 0.9 over 10 folds**, this confirms that a **Liquid State Machine** can handle **sparse**, **temporal**, and **highly noisy** data. The results also confirm the reservoir’s ability to **retain memory** of past inputs while processing new frames. This suggests the architecture is suitable for **online recognition** under jitter and noise, and that **increasing the dataset** can further improve accuracy.

---

## 🧠 Idea

- We generate a dataset of **32×32×100** videos (H×W×T) where a ball moves along different trajectories.
- Each video is fed to the LSM **frame by frame** (the simulator receives one frame at a time), and a set of **temporal features** is extracted from selected output neurons.
- A **readout** (Random Forest) learns to classify the orbit from the extracted temporal features.

---

## Script and usage pipeline

- **build_trajectory_dataset.py**: builds the videos and saves them with the correct label, ready to be fed into the reservoir. By default it generates **100 videos per trajectory**, but you can easily change this via the global variable `SAMPLES_PER_CLASS`. Generating **more examples per class** (i.e., a larger dataset) typically improves accuracy.
- **trajectory_viewer.py**: displays the generated videos, showing random examples of each trajectory in sequence.
- **build_reservoir_feature_dataset.py**: takes one video at a time, uses it as input to the reservoir, extracts features (`mean spike time`, `first spike time`, `last spike time`, `mean ISI`, `ISI variance`) from the output neurons (**35 neurons** randomly selected among the non-input ones), and saves them with the correct label so you can train the readout.
- **train_random_forest.py**: trains the readout (a Random Forest), prints **accuracy** on train/test, and draws the **confusion matrix**.
- **random_forest_cross_validation.py**: performs **10-fold cross-validation** of the Random Forest, returning **mean accuracy** and **mean confusion matrix**.

**Recommended pipeline**: after installing the dependencies from `requirements.txt`, run the scripts **in the order listed above**. `trajectory_viewer.py` is optional; it’s just a visual check to see the type of videos being used.

---

## 📦 Trajectory classes

- `left_right` – left to right  
- `right_left` – right to left  
- `top_bottom` – top to bottom  
- `bottom_top` – bottom to top  
- `clockwise` – circular, clockwise  
- `counter_clockwise` – circular, counter-clockwise  
- `random` – random positions

Each video has:
- **grid** 32×32,
- **100 frames**,
- ball with **random shape and size** (circle/ellipse),
- **position jitter** and **frame noise**,
- **random starting point**.

---

## 🔧 Requirements & installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv env
   # Linux/Mac
   source env/bin/activate
   # Windows
   .\env\Scripts\activate

 ```bash
 pip install -r requirements.txt

