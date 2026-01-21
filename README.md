# Optimal Cross-Layer Feature Selection with Genetic Algorithms for IoT/RPL Intrusion Detection

## Abstract

The Internet of Things (IoT) relies heavily on the Routing Protocol for Low-Power and Lossy Networks (RPL). However, RPL's inherent lack of robust security mechanisms makes it susceptible to severe routing attacks such as Hello Flood, Version Number, and Rank Attacks. Traditional Intrusion Detection Systems (IDS) often fail in this domain due to the high computational cost of processing high-dimensional traffic data on resource-constrained devices.

This repository implements a novel Hybrid IDS that integrates **Evolutionary Computing** and **GPU-Accelerated Machine Learning**. By employing Genetic Algorithms (GA), the system dynamically selects the optimal subset of Cross-Layer features, maximizing detection accuracy while minimizing computational overhead. The selected features are then used to train an XGBoost classifier accelerated via CUDA, achieving state-of-the-art performance suitable for Edge Computing scenarios.

## Key Contributions

1.  **Evolutionary Dimensionality Reduction:** Utilizes Genetic Algorithms (implemented via DEAP) to solve the feature selection problem as a multi-objective optimization task, balancing detection precision against feature count.
2.  **GPU-Accelerated Classification:** Leverages NVIDIA CUDA cores through XGBoost's histogram-based tree method (`tree_method='hist'`, `device='cuda'`), enabling ultra-fast training cycles and real-time inference capabilities.
3.  **Resilient Data Pipeline:** Features a custom-built raw data parser capable of recovering corrupted RPL traffic logs (Cooja simulation outputs) that standard libraries fail to process.
4.  **Cross-Layer Analysis:** Correlates metrics from Physical, MAC, and Network layers to identify complex attack patterns that evade single-layer detection.

## System Architecture

The project pipeline is structured into four sequential stages, as implemented in `CrossLayerGA.ipynb`:

### 1. Robust Data Ingestion & Preprocessing
Raw IoT traffic logs often contain structural errors (merged columns, unclosed quotes). We implemented a byte-level parser to reconstruct the tabular data.
* **Cleaning:** Handling of infinite values and hexadecimal artifacts.
* **Normalization:** Min-Max scaling to standardize feature ranges.
* **Encoding:** Label encoding for categorical protocol fields.

### 2. Evolutionary Feature Selection (The Genetic Algorithm)
We define the feature selection problem as finding a binary mask $M$ of length $N$ (total features).
* **Chromosome:** A binary vector where $1$ implies feature inclusion.
* **Fitness Function:** Designed to penalize complexity.
    $$Fitness = Accuracy - (\alpha \times \frac{\sum M}{N})$$
    Where $\alpha$ is a penalty coefficient (e.g., 0.05).
* **Operators:**
    * *Selection:* Tournament Selection (size=3).
    * *Crossover:* Two-Point Crossover.
    * *Mutation:* Bit-Flip Mutation (prob=0.05).

### 3. Model Training
The optimal feature subset identified by the GA is used to train an Extreme Gradient Boosting (XGBoost) model. The model is hyper-tuned for GPU execution, significantly reducing training time compared to CPU-based Random Forest implementations.

## Experimental Results

The system was evaluated using a stratified dataset containing both normal traffic and simulated RPL attacks.

### Performance Metrics
The model achieved the following performance on the test set:

| Metric | Score | Definition |
| :--- | :--- | :--- |
| **Accuracy** | **98.0%** | Ratio of correctly predicted observations. |
| **Precision** | **98.0%** | Ratio of correctly predicted positive observations. |
| **Recall** | **98.0%** | Ratio of correctly predicted positive observations to all actual positives. |
| **F1-Score** | **98.0%** | Weighted average of Precision and Recall. |
| **AUC** | **0.98** | Area Under the ROC Curve. |

### Visual Analysis
* **Confusion Matrix:** Shows near-zero False Positives, ensuring network availability is not compromised by false alarms.
* **ROC Curve:** Demonstrates high separability between classes effectively independent of the discrimination threshold.

## Comparison with State of the Art

This work addresses specific gaps identified in recent literature:

* **vs. Hafsa et al. (2025):** While Hafsa et al. achieved 99% accuracy using CatBoost, their approach relies on a static, manually selected set of 12 features. Our Evolutionary approach automates this process, allowing the system to adapt dynamically to new attack signatures without human intervention.
* **vs. Al Sawafi et al. (2023):** Their Deep Learning approach (CNN+LSTM) requires heavy computational resources. Our XGBoost+GA solution offers comparable precision (98%) with significantly lower latency and resource consumption, making it more viable for actual IoT hardware.
* **vs. Kim et al. (2024):** Unlike their Fuzzy Logic approach which is tailored specifically for Neighbor Suppression attacks, our ML-based solution is generalizable to multiple attack vectors including Rank and Version Number attacks.

## Installation and Usage

### Prerequisites
* Python 3.8 or higher
* NVIDIA GPU with CUDA support (Recommended for acceleration)

### Dependencies
Install the required libraries:
```bash
pip install numpy pandas scikit-learn xgboost deap matplotlib seaborn
