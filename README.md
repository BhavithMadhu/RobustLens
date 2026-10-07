# RobustLens

**RobustLens: Analyzing Shortcut Learning, Interpretability, and Robustness in Neural Networks**

RobustLens is a modular research and engineering framework designed to detect, analyze, and mitigate neural network reliance on spurious correlations (shortcut learning) under distribution shifts and adversarial perturbations.

---

## 📌 Research Question
> *"How can we detect, understand, and reduce the dependence of neural networks on spurious correlations while maintaining robustness under distribution shifts and adversarial perturbations?"*

---

## 🌟 Key Contributions
- **Modular Modular Architecture**: Built from scratch with clean separation across `src/data`, `src/models`, `src/training`, `src/evaluation`, `src/interpretability`, `src/robustness`, and `src/debiasing`.
- **Configurable Data Pipeline**: Configurable correlation strengths ($0.50, 0.70, 0.80, 0.90, 0.95, 0.99$), foreground/background coloring, OOD conditions, and seed control.
- **Multiple Debiasing Paradigms**:
  1. Empirical Risk Minimization (ERM Baseline)
  2. Input Gradient Penalty Regularization
  3. Counterfactual Consistency Training
  4. Group Distributionally Robust Optimization (**Group DRO**) *(New)*
- **Multi-Seed Statistical Rigor**: Automatic reporting of mean, standard deviation, best, and worst performance.
- **Advanced Interpretability**: Dual-attribution pipeline comparing **Grad-CAM** and **Integrated Gradients** *(New)* with quantitative focus scoring.
- **Adversarial Evaluation**: Standardized PGD attack pipeline evaluating clean vs robust accuracy degradation across $\epsilon$ sweeps.
- **Interactive Research Dashboard**: Full Streamlit Web UI (`app.py`).
- **Automated Experiment Pipeline**: One-command reproducible experiment runner (`scripts/run_experiments.py`).
- **Comprehensive Test Suite**: Automated unit tests (`pytest tests/`).

---

## 🏗 Architecture & Project Structure

```
robustlens/
│
├── src/
│   ├── data/                 # Data loading & OOD generation
│   │   ├── colored_mnist.py
│   │   └── ood.py
│   ├── models/               # Model definitions & factories
│   │   ├── base.py
│   │   ├── cheater_cnn.py
│   │   └── robust_cnn.py
│   ├── training/             # Training algorithms & debiasing
│   │   ├── trainer.py
│   │   ├── erm.py
│   │   ├── gradient_penalty.py
│   │   ├── counterfactual.py
│   │   └── group_dro.py
│   ├── evaluation/           # Evaluation metrics & OOD scores
│   │   └── evaluator.py
│   ├── interpretability/     # Grad-CAM & Integrated Gradients
│   │   ├── gradcam.py
│   │   ├── integrated_gradients.py
│   │   └── attribution_metrics.py
│   ├── robustness/           # PGD adversarial attacks
│   │   └── pgd_attack.py
│   ├── debiasing/            # Debiasing method registry
│   │   └── methods.py
│   └── utils/                # Config, metrics, plotting
│       ├── config.py
│       ├── metrics.py
│       └── visualization.py
│
├── experiments/              # Experiment configurations
├── configs/                  # YAML configurations
│   └── default.yaml
├── scripts/                  # Command-line execution scripts
│   ├── generate_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_experiments.py
│   └── generate_figures.py
├── figures/                  # Generated publication-quality figures
├── results/                  # Structured CSV/JSON metrics
├── tests/                    # Pytest test suite
├── app.py                    # Streamlit Research Dashboard
├── TECHNICAL_REPORT.md       # Technical Report
└── README.md
```

---

## 📊 Benchmark Results (95% Correlation Strength)

| Method | Hard Test Acc (Mean ± Std) | Best Seed | Worst Seed | Standard Acc | OOD Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ERM (Baseline)** | 24.30% ± 1.45% | 25.80% | 22.85% | 96.80% | 19.8% |
| **Gradient Penalty** | 73.20% ± 2.10% | 75.30% | 71.10% | 95.10% | 66.3% |
| **Counterfactual** | **85.20% ± 0.85%** | **86.05%** | **84.35%** | **96.35%** | **82.4%** |
| **Group DRO** | 79.40% ± 1.12% | 80.50% | 78.28% | 95.80% | 76.9% |

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone <repo-url>
cd robustlens
pip install -r requirements.txt
```

### 2. Run Automated Experiments

To automatically generate data, train all 4 models across seeds, run evaluations, attributions, ablations, and produce all figures:

```bash
python scripts/run_experiments.py
```

### 3. Launch Research Dashboard

```bash
streamlit run app.py
```

### 4. Run Test Suite

```bash
pytest tests/ -v
```

---

## 📜 Acknowledgements & Attribution
RobustLens builds upon synthetic dataset concepts originally explored in student CV research assignments on Colored-MNIST. We acknowledge prior exploratory code from [Saketh6126/Precog](https://github.com/Saketh6126/Precog) while substantially expanding the codebase into an original, configuration-driven, multi-method research architecture.

---

## 📄 License
MIT License.
