# 🧠 Sentiment Analysis using NLP and Machine Learning

## 📌 Overview
This project leverages **Natural Language Processing (NLP)** and **machine learning techniques** to classify customer reviews into sentiment categories: **Positive**, **Negative**, or **Neutral**. The main objective is to help businesses understand and act on customer feedback at scale by automating sentiment analysis.

---

## Business Objectives

- **Identify Customer Sentiment**: Determine the emotional tone behind customer reviews to guide product and service improvement.
- **Automate Feedback Processing**: Enable real-time classification of large volumes of textual data.
- **Improve Decision-Making**: Provide actionable insights to product, marketing, and customer service teams.
- **Highlight Product Issues**: Correlate sentiments with `product_category` to isolate underperforming or highly praised products.

---

##  Dataset Description

- **Text Data**: Customer reviews in raw text form.
- **Target**: Sentiment label (`Positive`, `Negative`, `Neutral`).
- **Additional Feature**: `product_category` column included for product-level sentiment tracking.

---

## 🔧 Workflow Summary

1. **Data Cleaning & Preprocessing**
   - Lowercasing, punctuation removal, stopword filtering, and lemmatization.
   - Class imbalance tackled using **SMOTE**.
  
2. **Feature Engineering**
   - Text converted using **TF-IDF Vectorization**.
   - `product_category` encoded via **OneHotEncoder** and combined with text features.

3. **Model Building & Evaluation**
   - Classifiers Used:
     - **Baseline Random Forest**
     - **Tuned Random Forest (GridSearchCV)**
     - **XGBoost (with and without SMOTE)**
     - **Multinomial Naive Bayes**
     - **Complement Naive Bayes**
   - Metrics:
     - **Accuracy**, **F1 Score**, **ROC AUC**
   - Model validation via **5-fold Cross-Validation**

4. **Visualization**
   - Sentiment Distribution Bar Chart
   - ROC AUC Score Comparison
   - Feature Importance Plot for Interpretability

---

## 🧪 Results Summary

| Model                      | Accuracy | F1 Score |
|---------------------------|----------|----------|
| Baseline RF               |   56.80% |   33.98  |
| Tuned RF (GridSearchCV)   |   66.89% |   53.65  |
| XGBoost (no SMOTE)        |   68.78% |   52.05  |
| XGBoost (with SMOTE)      |   67.79% |   53.18  |
| Multinomial Naive Bayes   |   67.79% |   53.18  |
| Complement Naive Bayes    |   62.81% |   54.81  |

- **Best F1 Score**: Complement Naive Bayes
- **Best Accuracy**: XGBoost without SMOTE

---

## 📈 Key Insights

- The **sentiment distribution** is imbalanced; SMOTE helped improve minority class representation.
- **Complement Naive Bayes** performed well despite its simplicity — a good option for lightweight deployment.
- **XGBoost** offered high accuracy but required careful handling of sparse data formats.
- Using `product_category` as a feature can help drill down into product-specific feedback for strategic improvements.

---

## Future Improvements

- Integrate deep learning models (e.g., LSTM, BERT) for better context understanding.
- Use advanced embeddings like Word2Vec or Sentence Transformers.
- Deploy the final model via a Flask/Django API or streamlit dashboard.
- Add topic modeling (e.g., LDA) for unsupervised insights into common review themes.

---

## 👨‍💼 Business Recommendations

- Monitor product-level sentiment to prioritize fixes or enhancements.
- Automate review screening to respond faster to negative feedback.
- Train customer service agents using real customer pain points.
- Feed model insights into marketing or user experience strategies.

---

## Dependencies

- `scikit-learn`
- `xgboost`
- `nltk`
- `imblearn`
- `matplotlib`
- `seaborn`
- `pandas`, `numpy`
- `scipy`

---


Project by: Wachira James
            Tim Musungu
            Vivian Kwamboka
            Clvin Mutua
            Hashim Ibrahim


