# CHE657_Proj

# Modelling Nuclear Fusion Reactor Power Outputs with Machine Learning Techniques

**CHE657 Course Project — Group 7**

> **Physics-informed and data-driven machine learning approaches for predicting nuclear fusion reactor power output**

## 👥 Authors

* **Chandan Achary** — 230319
* **Harshit Tomar** — 230462
* **Rutul Bhanushali** — 230884
* **Vishal Raj** — 231163

---

## 📌 Overview

Nuclear fusion is a promising source of clean and potentially abundant energy. However, accurately predicting fusion reactor performance is difficult because reactor behaviour depends on strongly coupled plasma, magnetic-confinement, fuel-composition, and operating parameters.

Experimental fusion data is also expensive and difficult to obtain at scale. This project investigates whether **machine learning can be used as a surrogate modelling approach for fusion reactor power prediction**, while also exploring whether incorporating simplified physical knowledge can improve model behaviour.

We compare three major approaches:

1. **Classical regression models**
2. **Data-driven neural networks**
3. **Physics-Informed Neural Networks (PINNs)**

The central idea is to compare a purely data-driven model against a model that attempts to incorporate a simplified fusion power relation:

$$
P(n,T)=\beta \Phi(T)n^2
$$

where:

* \(P\) = fusion power output
* \(n\) = fuel/plasma density
* \(T\) = plasma temperature
* \(\Phi(T)\) = temperature-dependent fusion reactivity term
* \(\beta\) = learned proportionality coefficient

The project ultimately demonstrates both the potential **and the limitations** of PINNs when the embedded physical model is itself an imperfect approximation.

---

# 🎯 Objectives

The project aims to:

* Build machine-learning models for predicting fusion reactor power output.
* Preprocess and numerically stabilize high-dynamic-range fusion data.
* Establish classical regression models as baselines.
* Develop a feed-forward neural-network regression model.
* Incorporate simplified fusion physics into a Physics-Informed Neural Network.
* Compare data-driven and physics-informed approaches.
* Investigate the effect of the physics-loss weighting parameter.
* Analyze why the current PINN formulation does not outperform the pure neural network.
* Identify future improvements to the physical model and learning methodology.

---

# 📊 Dataset

The implementation uses a simulated nuclear-fusion dataset containing **100,000 experimental configurations** and **19 columns**. The dataset represents simulated fusion experiments involving different reactor configurations and operating conditions.

### Major variables

The dataset contains parameters related to:

* Magnetic field fluctuations
* Leakage
* Plasma instabilities
* Magnetic field strength
* Magnetic field configuration
* Injection energy
* Beam symmetry
* Target density
* Target composition
* Fuel density
* Temperature
* Confinement time
* Fuel purity
* Energy input
* Power output
* Pressure
* Neutron yield
* Ignition

The target variable is:

```text
Power Output
```

### Dataset dimensions

```text
Samples: 100,000
Original features: 19
Numeric features: 16
Categorical features: 2
Final encoded features: 22
```

The categorical variables are:

```text
Magnetic Field Configuration
Target Composition
```

After one-hot encoding, the neural-network input contains **22 features**.

---

# ⚙️ Data Preprocessing

Fusion variables can span several orders of magnitude. For example, the implementation encounters temperatures on the order of:

$$
10^8\;K
$$

and densities on the order of:

$$
10^{19}-10^{20}\;m^{-3}
$$

Directly training neural networks on these quantities resulted in numerical instability, including occurrences of:

```text
β = NaN
```

Therefore, preprocessing was used to improve numerical stability.

### Preprocessing pipeline

```text
Raw Dataset
     │
     ▼
Remove invalid / missing entries
     │
     ▼
Categorical encoding
     │
     ▼
Numerical scaling
     │
     ▼
Train / Test Split
     │
     ├──────────────► Training Set
     │
     └──────────────► Test Set
```

### Scaling

The implementation uses the following characteristic scales:

| Quantity    |       Scale |
| ----------- | ----------: |
| Temperature |    \(10^8\) |
| Density     | \(10^{19}\) |
| Power       |    \(10^1\) |

The dataset is divided into:

```text
Training: 80,000 samples
Testing:  20,000 samples
```

Categorical variables are one-hot encoded, producing a final feature dimension of:

```text
22
```

---

# 🤖 Models

## 1. Linear Regression

Classical linear regression was used as a baseline.

Several regularization strategies were investigated:

* Ordinary Least Squares
* Ridge Regression
* Lasso Regression
* Elastic Net

Hyperparameters were explored using cross-validation and search-based optimization. The project ultimately found Elastic Net to be the preferred regularized linear formulation among the approaches investigated.

---

# 2. Data-Driven Neural Network

A feed-forward neural network was developed to learn the relationship between the reactor parameters and power output directly from data.

The model uses ReLU activations and an MLP-style architecture.

The project presentation describes the network as:

```text
64 → 32 → 16 → 1
```

while the final implementation used for the PINN experiments has the larger architecture:

```text
22 → 128 → 64 → 32 → 1
```

The implementation uses:

* PyTorch
* ReLU activations
* MSE loss
* AdamW optimizer
* Mini-batch training
* Hyperparameter search

The neural-network hyperparameters explored included:

```text
Learning rate: 1e-3, 5e-4
Weight decay:  1e-4, 1e-5
Batch size:    32, 64
Epochs:        100
```

The best configuration was selected based on validation MSE.

---

# 3. Physics-Informed Neural Network

The main focus of the project is the Physics-Informed Neural Network.

Rather than allowing the neural network to learn the entire relationship solely from data, a physics-based constraint is added to the loss function.

The total loss is:

$$
L=L_{\text{data}}+\lambda_{\text{phys}}L_{\text{physics}}
$$

where:

$$
L_{\text{data}}=\operatorname{MSE}(P_{\text{pred}},P_{\text{actual}})
$$

and:

$$
L_{\text{physics}}=\operatorname{MSE}\left(P_{\text{pred}},\beta n^2\Phi(T)\right)
$$

The implementation uses:

```text
λphys = 0.1
```

---

## 🧬 Fusion Physics Used

The simplified physical model is based on two major dependencies.

### Density dependence

Fusion reactions are treated as second-order processes with respect to particle density:

$$
P\propto n^2
$$

The intuition is that the reaction rate depends on collisions between particles, making density an important factor.

### Temperature dependence

The project uses a simplified Bosch–Hale-inspired reactivity relation:

$$
\Phi(T)
=
T^2
\exp\left(-\frac{b}{\sqrt{T}}\right)
$$

where \(b\) is a fitted parameter.

Combining the two gives:

$$
P(n,T)
=
\beta
T^2
\exp\left(-\frac{b}{\sqrt{T}}\right)
n^2
$$

This simplified relation was then incorporated into the PINN physics loss.

---

# 🧠 PINN Architecture

The implemented PINN uses:

```text
Input
  │
  ▼
22 features
  │
  ▼
128 neurons
  │
  ▼
64 neurons
  │
  ▼
32 neurons
  │
  ▼
1 output
```

Architecture:

```text
22 → 128 → 64 → 32 → 1
```

Training configuration:

```text
Epochs:          100
Physics weight:  0.1
Optimizer:       AdamW
Loss:            Data loss + Physics loss
```

The physics and data losses are simultaneously backpropagated during training.

---

# 🔬 Physics Parameter Estimation

The project attempted to estimate the parameters of the simplified physical relation from the available dataset.

The intended parameters were:

```text
β
b
```

However, the curve-fitting procedure encountered a broadcasting error:

```text
operands could not be broadcast together with shapes (2,) (100000,)
```

As a result, the implementation fell back to:

```text
βscaled = 1.0
b       = 20.0
```

with:

```text
βoriginal = 1.0 × 10^-53
```

This is an important limitation of the current implementation because the physics-loss equation is therefore based on a simplified and imperfectly fitted physical representation rather than a successfully optimized fusion-power law.

---

# 📈 Experimental Results

The final implementation compared three approaches:

1. Pure physics regression
2. Pure data-driven neural network
3. Physics-Informed Neural Network

### Test-set performance

| Model               |        MSE |        MAE |          R² |
| ------------------- | ---------: | ---------: | ----------: |
| Pure Physics Model  |     3307.7 |     49.848 |     -3.0190 |
| Pure Neural Network | **844.32** | **25.027** | **-0.0259** |
| PINN                |     856.06 |     25.132 |     -0.0401 |

The pure neural network achieved the lowest test MSE and the highest test \(R^2\) among the three implemented approaches.

---

# 🧪 What Did the Experiments Show?

An important result of this project is that **adding a physics loss did not automatically improve the model**.

The pure neural network achieved:

```text
Test MSE = 844.32
Test R²  = -0.0259
```

while the PINN achieved:

```text
Test MSE = 856.06
Test R²  = -0.0401
```

Thus, the PINN performed slightly worse than the pure data-driven network.

This was also visible in the loss curves: the physics component introduced significant noise into the total PINN loss. The project report explicitly identifies the physics-loss formulation as the likely source of this degradation.

This leads to an important conclusion:

> **A PINN is only as good as the physical constraints embedded into its loss function.**

If the physical relation is oversimplified or poorly fitted, enforcing it can actually hurt predictive performance.

---

# ⚖️ PINN vs Pure Neural Network

The project investigated different physics-loss configurations:

### Pure physics model

```text
Physics equation only
```

This performed poorly because the simplified physical relation does not capture all variables affecting the simulated reactor power output.

### Pure neural network

```text
λ = 0
```

The physics constraint is removed and the network learns entirely from data.

This produced the best predictive performance.

### PINN

```text
λ = 0.1
```

The physics constraint is introduced.

The result was slightly worse than the pure neural network, indicating that the current physical approximation is not sufficiently accurate to improve the model.

---

# 💡 Key Insights

### 1. Data preprocessing matters

The large numerical scales of temperature, density, and power caused unstable training. Appropriate normalization was necessary for stable neural-network training.

### 2. Neural networks captured relationships missed by simple regression

The classical physics-only and linear approaches performed substantially worse than the neural-network models.

### 3. Physics constraints are not automatically beneficial

The PINN did not outperform the pure neural network because the current physical relation is too simplified.

### 4. The choice of physics equation is critical

The current relation primarily captures:

$$
P\propto n^2\Phi(T)
$$

while other important reactor characteristics—including magnetic-field effects, confinement behaviour, fuel composition, and other reactor parameters—are not explicitly represented in the physics loss.

### 5. The project demonstrates a useful PINN failure mode

Rather than simply showing that PINNs work, this project demonstrates an important practical lesson:

**Incorrect or incomplete physics can make a physics-informed model worse than a purely data-driven model.**

---

# 🗂️ Project Structure

A suggested repository organization is:

```text
.
├── README.md
│
├── notebooks/
│   ├── che657_nuclear_fusion.ipynb
│   └── CHE657_Project_PINN_Implementation.ipynb
│
├── report/
│   ├── CHE657_Project_Report.pdf
│   └── CHE657_Group-7_Presentation.pdf
│
├── results/
│   └── Experiment_Summary.txt
│
└── data/
    └── fusion_dataset.csv
```

The notebooks contain the data-processing, model-training, physics-model, and PINN experiments.

---

# 🛠️ Technologies

The project primarily uses:

* **Python**
* **PyTorch**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Jupyter Notebook**

---

# ▶️ Running the Project

## 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

## 2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter
```

## 3. Launch Jupyter

```bash
jupyter notebook
```

## 4. Run the notebooks

The primary implementation notebooks are:

```text
che657_nuclear_fusion.ipynb
CHE657_Project_PINN_Implementation.ipynb
```

Run the preprocessing, model-training, and evaluation cells sequentially.

---

# 🔧 Important Configuration

The physics contribution to the PINN can be controlled through:

```python
PHYSICS_WEIGHT
```

The current experiment uses:

```python
PHYSICS_WEIGHT = 0.1
```

Changing this value allows investigation of the transition between:

```text
Pure Neural Network
        │
        │ increasing λ
        ▼
Physics-Informed Neural Network
```

---

# 🚀 Future Work

Several directions were identified for improving the project.

## 1. Symbolic Regression

Use symbolic regression to discover a more accurate relationship between reactor parameters and fusion power rather than imposing the current simplified equation.

## 2. Improved Physics Loss

Develop a richer physical model incorporating additional reactor parameters such as:

* Magnetic-field characteristics
* Confinement
* Fuel composition
* Plasma conditions
* Other experimentally relevant variables

## 3. Inverse PINN

Treat physical parameters such as \(b\) and \(\beta\) as trainable parameters within the PINN rather than estimating them separately.

## 4. Time-Dependent Modelling

Extend the model from steady-state prediction to time-dependent fusion reactor behaviour.

This could allow investigation of transient dynamics and the process of approaching steady-state operation.

## 5. Ensemble Models

Combine multiple models—including classical regression, neural networks, and PINNs—to investigate whether ensemble learning can improve robustness and prediction accuracy.

## 6. Improved Fusion Reactivity Representation

The current Bosch–Hale-inspired approximation is deliberately simplified. A more accurate representation of temperature-dependent fusion reactivity could potentially make the physics constraint substantially more useful.

---

# 🧪 Broader Chemical Engineering Applications

The same methodology can potentially be applied to other Chemical Engineering problems.

### Reactor Modelling

PINNs could combine reaction kinetics with plant data for systems such as:

* CSTRs
* PFRs
* Reaction networks

### Heat and Mass Transfer

Physics-informed models could incorporate governing transport equations for:

* Heat exchangers
* Catalyst beds
* Membranes
* Diffusion systems

### Process Control and Digital Twins

Physics-consistent machine-learning models could be used to build predictive models for process monitoring, control, and digital-twin applications.

### Kinetic Parameter Estimation

Unknown reaction parameters could be learned directly from limited experimental data.

### Multiphysics Systems

The approach could potentially be extended to coupled transport-reaction systems such as:

* CO₂ capture
* Electrochemical systems
* Coupled reaction and transport processes

---

# 📜 Project Context

This repository contains the implementation and documentation for the **CHE657 Course Project, Group 7**.

The project investigates the intersection of:

```text
Nuclear Fusion
      +
Chemical Engineering
      +
Machine Learning
      +
Deep Learning
      +
Physics-Informed Neural Networks
```

The primary takeaway is not simply that one model performs better than another, but that **successful physics-informed machine learning requires a sufficiently accurate representation of the underlying physical system**.

---

## ⭐ Key Result

> **The pure data-driven neural network currently provides the best predictive performance, while the PINN provides a framework for incorporating physical knowledge that could become more effective once the underlying fusion-power relation is improved.**

The current implementation therefore serves as a foundation for developing more sophisticated physics-informed models for fusion and other Chemical Engineering systems.

