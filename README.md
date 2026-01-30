# 📰 Fake News Detector (NLP + ML)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NLP](https://img.shields.io/badge/NLP-Text%20Processing-green)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

An end-to-end Natural Language Processing project that detects whether a news article is **Fake** or **Real** using Machine Learning. The system analyzes writing patterns using TF-IDF features and classifies articles through a trained Linear SVM model. A Streamlit web app provides real-time predictions.

---

## 🚀 Features
✔ Text preprocessing (cleaning, stopword removal, lemmatization)  
✔ TF-IDF vectorization  
✔ Model comparison (Logistic Regression, Naive Bayes, Linear SVM)  
✔ Best-performing model (Linear SVM) deployed  
✔ Interactive Streamlit interface  

---

## 🧠 Tech Stack
- Python  
- NLTK  
- Scikit-learn  
- Pandas & NumPy  
- Streamlit  

---

## 📊 Model Performance

| Model | Accuracy |
|------|----------|
| Naive Bayes | ~93% |
| Logistic Regression | ~98.9% |
| **Linear SVM** | **~99.4%** |

---

## 📂 Dataset
Fake News Detection Datasets (Kaggle):  
👉 https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

---

##⚠️ Disclaimer

This model classifies based on linguistic patterns, not factual verification. Predictions depend on patterns learned from the dataset.

## ▶ Installation & Run

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
streamlit run app.py
