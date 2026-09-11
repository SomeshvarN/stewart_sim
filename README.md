# 6-DOF Stewart Platform (Hexapod) Kinematics & Dexterity Simulator

A modular Python framework for modeling, animating, and analyzing a 6-Degree-of-Freedom (6-DOF) parallel manipulator (Stewart Platform / Hexapod). 

This repository provides full inverse kinematic modeling, analytical Jacobian evaluation, Newton-Raphson numerical forward kinematics, real-time trajectory visualization, and 3D workspace dexterity mapping.

---

## 🏗️ Project Architecture

The codebase is split into three clean, decoupled modules:

```text
stewart_sim/
├── platform_core.py        # Core Kinematics, Analytical Jacobian & FK Engine
├── simulation_dash.py      # Real-Time 3D Motion & Dexterity Dashboard
└── workspace_analyzer.py   # 3D Reachable Volume & Dexterity Heatmap Engine
