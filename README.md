# Airbnb-Success: Prediction of Review Activity & Segmentation

This project analyzes Airbnb listings across various neighborhoods to predict Review Activity and perform Unsupervised Segmentation. By integrating external datasets—including Crime, Population, and Walkability indices—this pipeline explores how local environment and property features influence guest engagement and listing performance.

---

## 📊 Data & Samples
To maintain a lightweight repository, this project includes **sample datasets (~100 rows each)**. These allow you to test the code logic without the overhead of the full 1.6 GB dataset.

**To run the full analysis:**
1. Download the full datasets from their original sources (Census Bureau, FBI, Inside Airbnb).
2. Create a local directory named `data/raw/` (this folder is ignored by Git).
3. Place the full files into the respective subfolders.
4. Update the file paths in the notebooks from `_sample` to the full data paths.

---

## Project Structure
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
|
├── src/                     # Python scripts for data processing
│   
├── notebooks/               # Jupyter notebooks for analysis
|   ├── Data Preprocessing + EDA.ipynb    # Data Cleaning and Merging
|   ├── Part A Supervised Learning.ipynb  # Linear Models & Regularization
│   ├── Part A2 KNN Trees.ipynb           # KNN and Tree-based Models
│   └── Part B Unsupervised Learning.ipynb # Clustering and Segmentation
│   
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

---

### Environment Setup

*   Python 3.12

*   Install required Python packages:

    `pip install -r requirements.txt`