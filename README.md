# Child Mind Institute — Problematic Internet Use 

## Description

This repository provides a predictive solution for the [Child Mind Institute - Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) competition. The goal is to estimate problematic internet use (PIU) in children and adolescents using biometric, demographic, psychological questionnaire data, and time-series sensor inputs.

It includes tabular feature processing, optional LLM-based text embeddings, and model training using XGBoost with stratified cross-validation.

---

## Folder Structure

```
├── problematic_internet_use.ipynb
├── requeriments.txt
```

---

## Installation

Create and activate a virtual environment, then install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Place the competition data** in the project directory.

   If you have the Kaggle API configured, run:
     ```
     kaggle competitions download -c child-mind-institute-problematic-internet-use
     ```

   OR you can download the data by clicking the folder above.

   [<span style="font-size:2em;">📁</span>](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data)

   At the bottom of the page you will find the download links for all files.
2. **Run the pipeline**:

   ```bash
   python main.py
   ```

---

## System Overview

The system consists of the following stages:

- **Data Loading**: Loads tabular and time-series data using `pandas` and `pyarrow`.
- **Feature Engineering**: Extracts biometric ratios, age buckets, and sensor-based metrics (e.g., enmo, light streaks).
- **Optional LLM Embedding**: Transforms psychometric responses into natural-language-like strings and encodes them into dense vectors via `SentenceTransformer`.
- **Model Training**: Applies `XGBoost` with Stratified K-Fold validation. Includes threshold optimization for ordinal classification.
- **Output Generation**: Creates a `submission.csv` file ready for Kaggle submission.

---

## Submission Format

The output `submission.csv` file will contain:

```
id,sii
00008ff9,3
000fd460,0
...
```

Where `sii` is the predicted score (integer between 0 and 3).

---

## Notes on LLM

If enabled, the system uses `SentenceTransformer` (MiniLM) to create semantic embeddings from questionnaire responses. This enhances model expressiveness by capturing latent psychological traits. The embeddings are integrated as numerical features alongside engineered data.

---

## References

- [Competition Overview](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview)
