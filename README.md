# 🛍️ SentimentIQ - AI Product Review Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

**SentimentIQ** is a state-of-the-art Machine Learning & Natural Language Processing (NLP) web application designed to analyze customer reviews for e-commerce products and predict sentiment in real-time. Built on a dataset of **30,000+ Amazon PC Product Reviews**, the platform leverages **TF-IDF Feature Extraction** and a **Linear Support Vector Machine Classifier (Linear SVC)** to deliver **92% prediction accuracy**.

---

## 🌟 Key Features

### 🔍 Real-Time Sentiment Predictor
- **Instant Inference:** Analyzes raw customer review text on the fly.
- **Confidence Scoring:** Outputs classification confidence percentage and decision boundary score.
- **Analysis Details Card:** Built-in collapsible breakdown displaying input text metrics, TF-IDF vectorization stats, and model prediction pipeline info without text overlapping or UI artifacts.

### 🏠 Interactive Dashboard & Analytics
- **Key Metrics Overview:** Total review count, positive sentiment rate (%), average star rating, and verified purchase ratio.
- **Visual Distributions:** Sentiment breakdowns (Positive vs. Negative) and Star Rating distributions (1 to 5 stars).
- **Monthly Trend Tracking:** Interactive area charts tracking customer review volume and sentiment trends over time.
- **Word Clouds:** Dual side-by-side cloud visualizers showing the most frequent keywords in positive vs. negative reviews.

### 📊 Dataset Explorer
- **Interactive Data Sampler:** Filter and inspect raw review records with adjustable row sample counts.
- **Distribution Deep-Dives:** Review length histograms, sentiment by star rating, and verified vs. non-verified purchase comparisons.
- **Correlation Heatmap:** Multi-feature correlation matrix (star rating, helpful votes, review length, sentiment score).

### 🧪 Model Evaluation & Benchmark
- **Multi-Model Benchmark:** Comparative accuracy analysis across 5 algorithms: **Linear SVC (92%)**, **Logistic Regression (91%)**, **SGD Classifier (90%)**, **Naive Bayes (87%)**, and **Random Forest (85%)**.
- **Confusion Matrix:** High-resolution confusion matrix heatmap for classification diagnostics.
- **Precision, Recall & F1-Score Gauges:** Visual metrics breakdown for positive (1) and negative (0) classes.

### 🎨 Modern UI / UX Design
- **Custom Dark Theme:** Modern glassmorphism UI with HSL gradient accents.
- **Responsive Navigation Drawer:** Custom floating left-aligned toggle button with animated hamburger-to-X icon transitions.

---

## 📊 Model Performance Summary

| Model | Accuracy | Precision (Pos) | Recall (Pos) | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVC** | **92.0%** | **0.93** | **0.97** | **0.95** | 🏆 **Selected** |
| **Logistic Regression** | 91.0% | 0.91 | 0.96 | 0.94 | Benchmark |
| **SGD Classifier** | 90.0% | 0.90 | 0.95 | 0.93 | Benchmark |
| **Naive Bayes (Multinomial)** | 87.0% | 0.88 | 0.92 | 0.90 | Benchmark |
| **Random Forest** | 85.0% | 0.86 | 0.91 | 0.88 | Benchmark |

---

## 📁 Project Structure

```text
Product-Sentiment-Analysis-E-commerce-Reviews/
│
├── sentiment_dashboard.py       # Main Streamlit web application & UI
├── Product_Review_Analysis.ipynb# Jupyter Notebook for EDA & model training
├── Amazon Product Review.txt    # Amazon Customer Reviews Dataset (~30k rows)
├── sentiment_model.pkl          # Trained Linear SVC Machine Learning model
├── vectorizer.pkl               # Fitted TF-IDF Vectorizer (5,000 features)
├── dashboard.png.png            # Application preview screenshot
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

---

## 🛠️ Tech Stack & Technologies

- **Language:** Python 3.8+
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (LinearSVC, TF-IDF Vectorizer, Joblib)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [WordCloud](https://github.com/amueller/word_cloud)
- **Front-End Styling:** Custom CSS, Glassmorphism UI, Responsive Flexbox & Grid layouts

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.8 or higher is installed on your system.
Verify installation by running:
```bash
python --version
```

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/Product-Sentiment-Analysis-E-commerce-Reviews.git
cd Product-Sentiment-Analysis-E-commerce-Reviews
```

### 3. Install Dependencies
Install all required Python packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
Launch the dashboard locally:
```bash
python -m streamlit run sentiment_dashboard.py
```
*Alternatively:*
```bash
streamlit run sentiment_dashboard.py
```

The app will launch in your default browser at:
`http://localhost:8501`

---

## 📖 Methodology & Pipeline

1. **Data Ingestion:** Loaded 30,846 Amazon PC product reviews.
2. **Text Preprocessing:** Cleaned HTML markup, lowercased text, removed special characters, and handled missing values.
3. **Sentiment Binarization:** Labeled reviews with 4-5 stars as **Positive (1)** and 1-2 stars as **Negative (0)**.
4. **Vectorization:** Applied `TfidfVectorizer` with `max_features=5000` to capture term frequency-inverse document frequency.
5. **Model Training & Benchmark:** Trained Linear SVC, Logistic Regression, Naive Bayes, Random Forest, and SGD Classifiers.
6. **Deployment:** Serialized the best performing model (`sentiment_model.pkl`) and vectorizer (`vectorizer.pkl`) with Joblib and deployed via Streamlit.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
