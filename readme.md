# GPU Cluster Orchestration Demo

A simplified demonstration project showcasing basic concepts of GPU cluster orchestration, job scheduling, monitoring, and GPU programming principles.

## Project Overview

This project demonstrates understanding of key technologies used in GPU cluster management:

1. **Job Scheduling** (similar to Slurm concepts)
2. **Resource Monitoring** (conceptually similar to Prometheus/Grafana)
3. **GPU Programming** (CUDA concepts without requiring compilation)
4. **Distributed Computing** principles

## Components

### 1. Cluster Scheduler Simulation

A Python-based simulation of a GPU cluster scheduler that demonstrates:
* Priority-based job scheduling
* Resource allocation across multiple nodes
* Job queue management
* Utilization tracking

### 2. GPU Programming Concepts

Two Python files demonstrating GPU concepts:
* `gpu_simulation.py`: Simulates and visualizes CPU vs GPU performance
* `pseudo_cuda_explanation.py`: Educational explanation of CUDA programming concepts

### 3. Example Configurations

Sample configuration files showing familiarity with:
* Slurm job scripts (examples of how to configure GPU jobs)
* Prometheus configuration (example of monitoring setup)

## How to Run

1. Install dependencies:
```
pip install numpy matplotlib
```

2. scheduler simulation:
```
python simple_scheduler.py
```

3. GPU performance simulation:
```
python gpu_simulation.py
```

4. CUDA programming concepts:
```
python pseudo_cuda_explanation.py
```

## Simulation Results and Visualizations
<img width="872" alt="Screenshot 2025-03-25 at 15 31 11" src="https://github.com/user-attachments/assets/1bd3b83d-9faa-49c2-bf34-6b90d0fd57ca" />
This visualization shows simulated GPU cluster usage over time. The top graph tracks overall utilization, declining from 62% to 0% with a brief spike midway. The bottom graph breaks down activity by node, revealing:

- Node 0: Most actively used, with periods of 100% and 75% utilization
- Node 1 & Node 2: Moderate usage (75%) with staggered job completions 
- Node 3: Completely unused throughout

This demonstrates dynamic resource allocation across a cluster as jobs are scheduled and completed, highlighting potential load balancing issues with Node 3 sitting idle.


