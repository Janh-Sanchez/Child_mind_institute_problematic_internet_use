# Problematic Internet Use Prediction

## Description

This repository contains a solution for the [Child Mind Institute - Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) Kaggle competition. The objective is to predict the severity of problematic internet use in children and adolescents using both questionnaire data and wearable sensor time series.

The workflow includes data preprocessing, feature engineering, model training (using LightGBM), and prediction generation for submission.

---

## Folder Structure

- `main.py`: Main script for data processing, feature extraction, modeling, and prediction.
- `requirements.txt`: List of required Python packages.
- `train.csv`, `test.csv`, `sample_submission.csv`: Tabular data files from the competition (See [Usage](#usage) section) .
- `series_train.parquet`, `series_test.parquet`: Parquet files with time series sensor data (See [Usage](#usage) section) .
- `submission.csv`: Output file ready for Kaggle submission (generated after running the pipeline).


---

## Installation

Make sure you have Python 3.7 or higher installed.

Install all dependencies with:

```
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy
pandas
matplotlib
tqdm
scipy
scikit-learn
lightgbm
optuna
pyarrow
```

---

## Usage

1. **Download the competition data** and place all files in the `data/` folder.

   [📁](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data)

   - You can download the data by clicking the folder above.
   - At the bottom of the page you will find the download links for all files.
   - Or, if you have the Kaggle API configured, run:
     ```
     kaggle competitions download -c child-mind-institute-problematic-internet-use
     ```

2. **Run the main script** from the project root:

   ```
   python main.py
   ```

3. **Result:**  
   The script will generate a `submission.csv` file ready for submission to Kaggle.

---

## System Overview

- **Data Loading:** Reads and merges questionnaire and time series data.
- **Preprocessing:** Handles missing values, encodes categorical variables, and extracts features from time series.
- **Feature Engineering:** Creates new features from both sensor and questionnaire data.
- **Model Training:** Trains a LightGBM model using cross-validation and optimizes thresholds for ordinal prediction.
- **Prediction:** Generates predictions for the test set and formats them for submission.

---

## Submission Format

The output file `submission.csv` will have the following format:

```
id,sii
00008ff9,3
000fd460,0
...
```
- `id`: Identifier for each test sample.
- `sii`: Predicted SII category (0 to 3).

---

## References

- [Kaggle Competition Overview](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview)

---

## Group

- **Jan Henrik Sánchez Jerez** – 20231020130  
- **Juan David Castaño González** – 20231020131  
