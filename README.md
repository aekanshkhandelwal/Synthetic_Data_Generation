# Fraud Detection Synthetic Data Generation (Generator–Discriminator)

This project implements a pipeline to generate synthetic fraud transaction data using a Conditional GAN (CTGAN). The goal is to augment datasets with realistic synthetic samples, particularly focusing on fraudulent transactions (`is_scam = 1`).

## Project Overview

The core of this project is a Jupyter Notebook that performs the following steps:
1.  **Setup & Data Loading**: Installs necessary libraries and loads the `crypto_scam_transaction_dataset.csv`.
2.  **Data Preprocessing**:
    - Handles missing values (imputation).
    - Converts timestamps to usable features (hour, day of week, time gap).
    - Scales numerical features using MinMax scaling.
3.  **Model Training**: Trains a CTGAN (Conditional Tabular GAN) on the preprocessed data.
4.  **Synthetic Data Generation**: Generates synthetic samples, specifically targeting scam transactions.
5.  **Validation & Export**: Combines real and synthetic data, validates distributions, and exports the final dataset to `synthetic_fraud_dataset.csv`.

## Project Workflow

The following diagram illustrates the end-to-end pipeline for generating synthetic fraud data:

![Project Workflow](workflow.png)

```mermaid
graph TD
    A[Raw Dataset: crypto_scam_transaction_dataset.csv] --> B[Data Preprocessing]
    B --> B1[Missing Value Imputation]
    B --> B2[Timestamp conversion: Hour, DayOfWeek, TimeGap]
    B --> B3[Numerical Feature Scaling: MinMax]
    B3 --> C[CTGAN Training]
    subgraph CTGAN Process
    C1[Generator: Learns distribution from Conditional Vector]
    C2[Discriminator: Distinguishes Real vs Synthetic]
    end
    C --> C1
    C1 --> C2
    C2 --> D[Conditional Sampling: targeting is_scam=1]
    D --> E[Validation & Quality Assessment]
    E --> F[Output: synthetic_fraud_dataset.csv]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#00ff00,stroke:#333,stroke-width:2px
    style CTGAN Process fill:#e1f5fe,stroke:#01579b,stroke-dasharray: 5 5
```

## CTGAN Architecture

This project utilizes Meta's **CTGAN (Conditional Tabular GAN)**, specialized for tabular data. It addresses common GAN challenges in tabular domains like non-Gaussian distributions and imbalanced categorical features.

### Core Components:
- **Generator**: A deep neural network that takes a noise vector and a conditional vector as input to produce synthetic rows.
- **Discriminator**: Evaluates synthetic samples against real samples, providing feedback to the generator.
- **Mode-Specific Normalization**: Handles complex numerical distributions by using Variational Gaussian Mixture models.
- **Conditional Vector & Training-by-Sampling**: Ensures that all categories in discrete columns are sampled evenly during training.

```mermaid
graph LR
    subgraph Architecture
    NV[Noise Vector] --> G[Generator]
    CV[Conditional Vector] --> G
    G --> SD[Synthetic Data]
    RD[Real Data] --> MSN[Mode-Specific Normalization]
    MSN --> D[Discriminator]
    SD --> D
    D --> L[Loss/Feedback]
    L --> G
    L --> D
    end
    
    style G fill:#bbdefb,stroke:#1976d2
    style D fill:#ffccbc,stroke:#e64a19
    style L fill:#c8e6c9,stroke:#388e3c
```

## Files

- `notebook.ipynb`: The main Jupyter Notebook containing the entire pipeline.
- `crypto_scam_transaction_dataset.csv`: The source dataset containing real transaction data.
- `synthetic_fraud_dataset.csv`: The generated output dataset (created after running the notebook).

## Dependencies

To run this project, you need the following Python libraries:

- `ctgan`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install them via pip:

```bash
pip install ctgan pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1.  Ensure you have the required dependencies installed.
2.  Open `notebook.ipynb` in Jupyter Notebook or JupyterLab.
3.  Run all cells sequentially.
4.  The script will train the model and generate a new file named `synthetic_fraud_dataset.csv` in the same directory.