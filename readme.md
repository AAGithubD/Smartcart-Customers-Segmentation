# 🛒 SmartCart – Customer Segmentation using Unsupervised Machine Learning

SmartCart is an end-to-end **customer segmentation** project that groups 2,240 retail customers into distinct behavioural segments using **PCA, K-Means and Agglomerative (Hierarchical) Clustering**. The resulting segments help a retailer understand *who* its customers are, *how* they shop, and *which* groups to target with campaigns.

---

## 📌 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Dataset](#-dataset)
3. [Tech Stack](#-tech-stack)
4. [Project Workflow](#-project-workflow)
5. [Key Results](#-key-results)
6. [Business Recommendations](#-business-recommendations)
7. [Project Structure](#-project-structure)

---

## 🎯 Problem Statement
Treating every customer the same wastes marketing budget. SmartCart needs to identify natural groups of customers based on demographics, income, purchasing channels and spending, so that promotions, retention and product strategies can be personalised per segment.

**Objective:** Build an unsupervised learning pipeline that segments customers and profiles each segment in a business-friendly way.

---

## 📂 Dataset
File: `smartcart_customers.csv` — **2,240 rows × 22 columns**

| Category | Columns |
|---|---|
| Demographics | `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Customer relationship | `Dt_Customer` (enrolment date), `Recency` (days since last purchase), `Complain` |
| Spending (last 2 yrs) | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Purchase behaviour | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Campaign | `Response` (accepted last campaign) |

Data quality: only `Income` had missing values (24 rows).

---

## 🛠 Tech Stack
- **Language:** Python 3
- **Data handling:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn (`OneHotEncoder`, `StandardScaler`, `PCA`, `KMeans`, `AgglomerativeClustering`, `silhouette_score`)
- **K selection:** KneeLocator (`kneed`)
- **Environment:** Jupyter Notebook

---

## 🔄 Project Workflow

### 1. Data Preprocessing
- Imputed 24 missing `Income` values with the **median** (robust to skew).

### 2. Feature Engineering
| New feature | Logic |
|---|---|
| `Age` | Current year − `Year_Birth` |
| `Customer_Tenure_Days` | Days between enrolment and the latest enrolment date in the data |
| `Total_Spending` | Sum of all six product-category spends |
| `Total_Children` | `Kidhome` + `Teenhome` |
| `Education` (grouped) | Basic / 2n Cycle → *Undergraduate*; Graduation → *Graduate*; Master / PhD → *Postgraduate* |
| `Living_With` | Married / Together → *Partner*; Single / Divorced / Widow / Absurd / YOLO → *Alone* |

Redundant columns (`ID`, `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer` and individual `Mnt*` columns) were dropped, leaving 15 features.

### 3. Outlier Removal
Pair plots revealed extreme values. Customers with **Age ≥ 90** or **Income ≥ 600,000** were removed (2,240 → **2,236** rows).

### 4. Exploratory Analysis
- Pair plots of Income, Recency, Response, Age, Total Spending, Total Children
- Correlation heatmap of numeric features

### 5. Encoding & Scaling
- One-hot encoding of `Education` and `Living_With`
- `StandardScaler` so that all 18 features contribute equally to distance calculations

### 6. Dimensionality Reduction (PCA)
- Reduced to **3 principal components** for clustering and 3D visualisation
- Explained variance: PC1 ≈ 23.2 %, PC2 ≈ 11.4 %, PC3 ≈ 10.4 %

### 7. Choosing the Number of Clusters
- **Elbow method** (WCSS) with automatic knee detection → **k = 4**
- **Silhouette score** analysed for k = 2…10 and plotted alongside WCSS to validate the choice

### 8. Clustering
- **K-Means** (k = 4, `random_state=42`)
- **Agglomerative clustering** (k = 4, Ward linkage) — used for the final profiling
- Both visualised in 3D PCA space

### 9. Cluster Profiling
Cluster sizes, income vs. spending scatter plots and per-cluster feature means were used to interpret each segment.

---

## 📊 Key Results

Four segments emerged. Two dimensions separate them most clearly: **spending / income level** (high vs. low) and **household type** (living with a partner vs. alone).

| Cluster | Profile | Avg. Income | Avg. Total Spend | Avg. Children | Living | Campaign Response |
|:--:|---|---:|---:|---:|---|---:|
| 0 | Low-spend, family households | ~$39.7K | ~222 | 1.24 | Partner | 7.6 % |
| 1 | **High-value** partnered customers | ~$72.8K | ~1,237 | 0.51 | Partner | 16.7 % |
| 2 | Low-spend, single customers | ~$37.0K | ~166 | 1.27 | Alone | 14.2 % |
| 3 | **High-value** single customers | ~$70.7K | ~1,190 | 0.46 | Alone | **32.0 %** |

**Insights**
- High-income customers (clusters 1 & 3) spend roughly **5–7× more** than low-income ones and shop far more via **catalog (~5 purchases)** and **store (~8–9 purchases)**.
- Low-income clusters (0 & 2) visit the website more (~6.3–6.7 visits/month) but buy little, and rely more on **deals** — they are price-sensitive browsers.
- Cluster 3 (high-income singles) responds to campaigns at **~2× the rate** of any other group, making it the best campaign target.
- High-value customers tend to have fewer children (~0.5 vs ~1.25) and are slightly older (~59 vs ~56).
- Recency is nearly identical across clusters (~48–50 days), so it is not a differentiator.

---

## 💡 Business Recommendations
- **Clusters 1 & 3 (high-value):** loyalty programmes, premium/early-access offers, catalog and in-store personalisation; prioritise cluster 3 for campaigns.
- **Clusters 0 & 2 (low-spend, deal-seeking):** discount bundles, web-exclusive coupons and family packs to convert high web traffic into purchases.
- **Cluster 0 (family households):** family-size and value bundles.
- **Cluster 2 (singles):** smaller pack sizes and convenience products.

---

## 📁 Project Structure
```
smartcart/
├── smartcart.ipynb            # Full analysis notebook
├── smartcart_customers.csv    # Dataset
├── requirements.txt
└── README.md
```

---



⭐ If you found this project useful, consider giving it a star!
