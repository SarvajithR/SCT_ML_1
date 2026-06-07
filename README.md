# 🏠 House Price Prediction — Linear Regression

> **SkillCraft Technology ML Internship | Task 01**

Predicts house prices based on square footage, number of bedrooms, and number of bathrooms using Multiple Linear Regression.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9994 (99.94%) |
| **MAE** | ₹1,973.29 |
| **RMSE** | ₹2,844.27 |
| Train / Test Split | 80% / 20% |
| Dataset Size | 50 samples |

---

## 🗂️ Project Structure

```
House-Price-Prediction-Linear-Regression/
│
├── predict_house_prices.ipynb   # Full notebook with EDA + model
├── predict_house_prices.py      # Standalone Python script
├── house_prices.csv             # Dataset
├── requirements.txt             # Dependencies
│
└── images/
    ├── heatmap.png              # Feature correlation heatmap
    ├── distribution.png         # Price distribution plot
    ├── prediction.png           # Actual vs Predicted scatter
    └── metrics.png              # Evaluation metrics summary
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction-Linear-Regression.git
cd House-Price-Prediction-Linear-Regression

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the script
python predict_house_prices.py
```

Or open `predict_house_prices.ipynb` in Jupyter / Google Colab.

---

## 🔍 Features Used

| Feature | Description |
|---------|-------------|
| `SquareFootage` | Total area of the house (sq ft) |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |

**Target:** `Price` — House sale price in USD

---

## 🧰 Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-Learn` · `Matplotlib` · `Seaborn`

---

## 📈 Visualizations

**Correlation Heatmap**


**Actual vs Predicted**


**Model Metrics**


---

## 💡 Key Learnings

- Implemented Multiple Linear Regression from scratch using Scikit-Learn
- Performed EDA: summary stats, distributions, and correlation analysis
- Evaluated model with MAE, RMSE, and R² metrics
- Built an interactive prediction system via CLI
- Visualized results with Matplotlib and Seaborn

---

*Part of the SkillCraft Technology Machine Learning Internship Program*
