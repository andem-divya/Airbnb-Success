# Airbnb-Success: Prediction of Review Activity & Segmentation

This project analyzes Airbnb listings across various neighborhoods to predict Review Activity and perform Unsupervised Segmentation. By integrating external datasets—including Crime, Population, and Walkability indices—this pipeline explores how local environment and property features influence guest engagement and listing performance.

---

## Project Structure
```text
├── data/
│   ├── raw/                 # Original unmodified datasets
│   │   ├── airbnb/          # Airbnb listings data
│   │   ├── census_tract/    # Census tracts shapefiles
│   │   ├── crime/           # Local crime statistics
│   │   ├── population/      # Population density data
│   │   └── walkability/     # National Walkability Index data
│   └── preprocessed/        # Cleaned and feature-engineered datasets

|
├── src/                     # Python scripts for data processing and utility functions
│   
├── notebooks/               # Jupyter notebooks for analysis
|   ├── Data Preprocessing + EDA.ipynb    # Data Cleaning, Merging, and Exploratory Analysis
│   ├── Part A Supervised Learning.ipynb  # Supervised Learning: Linear Regression & Regularization
│   ├── Part A2 KNN Trees.ipynb           # Supervised Learning: KNN and Tree-based Models
│   ├── Part B Unsupervised Learning.ipynb # Unsupervised Learning: Clustering and Segmentation
│   
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
