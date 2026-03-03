# Airbnb-Success: Prediction of Review Activity & Segmentation

This project analyzes Airbnb listings across various neighborhoods to predict Review Activity and perform Unsupervised Segmentation. By integrating external datasets—including Crime, Population, and Walkability indices—this pipeline explores how local environment and property features influence guest engagement and listing performance.

---

## 📊 Data & Samples
To maintain a lightweight repository, this project includes **sample datasets (~100 rows each)**. These allow you to test the code logic without the overhead of the full 1.6 GB dataset.

### 📦 Full Dataset

Due to GitHub size limitations, the complete dataset is hosted externally.

**Download full structured dataset (Google Drive):**  
[Download Full Dataset](https://drive.google.com/drive/folders/1aXCUeh5X_QFl0poqNo8bH_zcg-qRQBDp?usp=sharing)

The Google Drive folder already mirrors the expected project structure:

```text
data/
├── raw/
├── preprocessed/
```

### To Run the Full Analysis

1. Download the Google Drive folder.
2. Place the entire `data/` folder into the project root directory.
3. Ensure it sits alongside `src/`, `notebooks/`, and `README.md`.
4. Run the notebooks as usual.

---

## 📁 Project Structure
```text
├── data/
│   ├── raw_sample/          # SAMPLE DATA (100 rows) - Pushed to GitHub
│   │   ├── airbnb/          # Airbnb listings sample
│   │   ├── census_tract/    # Census tracts shapefiles sample
│   │   ├── crime/           # Local crime statistics sample
│   │   ├── population/      # Population density sample
│   │   └── walkability/     # National Walkability Index sample
│   └── preprocessed_sample/ # SAMPLE DATA - Feature-engineered outputs
│
├── src/                     # Python scripts for data processing
├── notebooks/               # Jupyter notebooks for analysis
│   ├── Data Preprocessing + EDA.ipynb    # Data Cleaning and Merging
│   ├── Part A Supervised Learning.ipynb  # Linear Models & Regularization
│   ├── Part A2 KNN Trees.ipynb           # KNN and Tree-based Models
│   └── Part B Unsupervised Learning.ipynb # Clustering and Segmentation
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── Milestone 2 Final Report Airbnb Success.pdf       # Final project report / analysis
```

---

## ⚙️ Environment Setup

* Python 3.12

* Install required Python packages:

```
pip install -r requirements.txt
```

## Authors
* Divya Andem - divyaand@umich.edu
* Jordan Huang - jordanhu@umich.edu
* Sophia Settle - sosettle@umich.edu