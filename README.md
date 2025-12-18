# Interpretable Reinforcement Learning for Supply Chain Optimization


**Authors:**

* Zhengan Du
* Tianqi Jin
* Zijun Meng
* Kaiwen Yang

This repository contains the code and experiments for the **MIE1666 – Machine Learning in Mathematical Optimization** course project at the University of Toronto, conducted under the supervision of Professor Elias Khalil.

The project investigates **explanable reinforcement learning (XRL)** methods for solving **multi-echelon supply chain network problems with backlogs**, using the **NetworkManagement-v0** environment from **OR-Gym**.

Our goal is to bridge the gap between:

* high-performance but opaque **black-box reinforcement learning** models, and
* **inherently interpretable additive models** that enable transparent, auditable decision-making in operational settings.

---

## Project Overview

Multi-echelon supply chains involve sequential decisions across suppliers, producers, distributors, and retailers under stochastic demand, lead times, and capacity constraints. Classical optimization approaches (e.g., deterministic LPs and multi-stage stochastic programs) provide strong baselines but often struggle with scalability and non-stationarity. Reinforcement learning (RL) offers adaptability, but learned policies are typically black boxes.

This project studies:

* a **black-box PPO (Proximal Policy Optimization)** agent as a performance benchmark, and
* **interpretable policy models** based on the **Generalized Additive Model (GAM)** family, including neural additive variants.

The central research question is:

**Can interpretable reinforcement learning policies achieve competitive performance on multi-echelon supply chain control problems while providing transparent, human-understandable decision rules?**

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── report.pdf
│
├── mlp.ipynb                 # Black-box PPO (MLP policy) experiments
├── dynamite.ipynb            # Interpretable additive policy experiments
├── nam_test32_tuning.ipynb   # Neural Additive Model (NAM / NODE-GAM) tuning
├── node_vis.ipynb            # Visualization of learned additive components
```

All experiments are implemented as **Jupyter notebooks** to emphasize clarity, inspection, and reproducibility.

---

## Environment and Dependencies

### Python Version

* Python **3.8** (required due to OR-Gym and RLlib compatibility).

### Key Libraries

* `or-gym` – supply chain reinforcement learning environments
* `ray[rllib,tune]==1.0.0` – PPO training and hyperparameter tuning
* `tensorflow==2.3.0`
* `nodegam` – neural generalized additive models
* `numpy`, `pandas`, `scipy`, `matplotlib`

The full dependency list is provided in `requirements.txt`.

### Setup Instructions

We strongly recommend using a virtual environment:

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note: OR-Gym depends on older versions of several libraries. Installing inside a clean environment is strongly advised.

---

## Supply Chain Environment

All experiments are conducted using **NetworkManagement-v0**, a reinforcement learning environment for **multi-echelon supply chain networks with backlogs**.

The environment models a four-echelon system:

* raw material suppliers
* producers
* distributors
* retailers

Key features include:

* stochastic customer demand (Poisson)
* inventory holding costs
* backlog penalties
* production and transportation constraints
* pipeline (in-transit) inventory with lead times

The environment generates fully synthetic but realistic time-series data, enabling controlled and reproducible experimentation.

---

## Methods

### 1. Black-Box Reinforcement Learning (PPO)

We implement a **Proximal Policy Optimization (PPO)** agent with a multi-layer perceptron (MLP) policy as a strong black-box baseline.

* State: on-hand inventory, pipeline inventory, observed demand
* Action: replenishment orders to upstream nodes
* Reward: total system profit (sales minus production, holding, operating, and backlog costs)

This model establishes a **performance upper bound** against which interpretable policies are evaluated.

Relevant notebook:

* `mlp.ipynb`

---

### 2. Interpretable Reinforcement Learning (GAM / NAM)

To achieve interpretability, we replace opaque neural policies with **additive models**, where each input feature contributes independently to the decision.

We explore:

* Generalized Additive Models (GAMs)
* Neural Additive Models (NAMs)
  * Discrete Neural Additive (DNAMite) Model
  * Neural Oblivious Decision Trees (NODE) GAM

These models enable:

* direct visualization of feature-action relationships
* transparent ordering strategies
* improved trust and auditability for operational decision-making

Relevant notebooks:

* `dynamite.ipynb`
* `nam_test32_tuning.ipynb`

---

## Analysis and Visualization

Beyond aggregate reward metrics, we analyze:

* learned additive shape functions for each state variable
* monotonicity and threshold behavior in replenishment decisions
* qualitative differences between black-box and interpretable policies

The notebook `node_vis.ipynb` focuses on extracting **human-interpretable decision logic**, rather than relying on post-hoc explanation tools.

---

## Baselines

Performance is benchmarked against classical operations research models reported in prior work:

* Deterministic Linear Programming (DLP)
* Multi-Stage Stochastic Programming (MSSP)

Rather than re-implementing these models, we use **published benchmark results** as non-learning performance targets. This allows us to focus on interpretability–performance trade-offs in learning-based methods.

---

## Reproducibility Notes

* Random seeds are fixed where applicable
* All experiments run on CPU (GPU not required)
* Results are reproducible under the specified dependency versions

Due to legacy dependencies, results may vary if newer library versions are used.

---

## Citations

If you use this repository or build upon it, please cite the following works.

### OR-Gym and Network Management Environment

```bibtex
@misc{HubbsORGym,
  author = {Christian D. Hubbs and Hector D. Perez and Owais Sarwar and Nikolaos V. Sahinidis},
  title  = {OR-Gym: A Reinforcement Learning Library for Operations Research Problems},
  year   = {2020},
  eprint = {arXiv:2008.06319}
}

@article{AlKahtaniNetworkManagement,
  author  = {Al-Kahtani, Mohammed and Alnajem, Mohammad and Alharkan, Ibrahim},
  title   = {Reinforcement Learning Environment for Multi-Echelon Supply Chain Networks with Backlogs},
  journal = {Processes},
  volume  = {9},
  number  = {1},
  pages   = {102},
  year    = {2021},
  doi     = {10.3390/pr9010102}
}
```

### Interpretable and Additive Models

```bibtex
@article{HastieTibshiraniGAM,
  author  = {Hastie, Trevor J. and Tibshirani, Robert},
  title   = {Generalized Additive Models},
  journal = {Statistical Science},
  volume  = {1},
  number  = {3},
  pages   = {297--310},
  year    = {1986}
}

@misc{ChangNODEGAM,
  author = {Chang, Chun-Hao and Caruana, Rich and Goldenberg, Anna},
  title  = {NODE-GAM: Neural Generalized Additive Models for Interpretable Deep Learning},
  year   = {2021},
  eprint = {arXiv:2106.01613}
}

@misc{NoriInterpretML,
  author = {Nori, Harsha and Jenkins, Samuel and Koch, Paul and Caruana, Rich},
  title  = {InterpretML: A Unified Framework for Machine Learning Interpretability},
  year   = {2019},
  eprint = {arXiv:1909.09223}
}

@article{RudinStopExplaining,
  author  = {Rudin, Cynthia},
  title   = {Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead},
  journal = {Nature Machine Intelligence},
  volume  = {1},
  number  = {5},
  pages   = {206--215},
  year    = {2019},
  doi     = {10.1038/s42256-019-0048-x}
}
```


