## 📌 Project Overview
Misinformation is a significant issue in today's digital age. This project is a **Natural Language Processing (NLP)** solution designed to detect and classify news articles as **REAL** or **FAKE**. 

Using a dataset of over 40,000 articles, I built an end-to-end Machine Learning pipeline that processes unstructured text data and predicts its authenticity with high precision using the **Multinomial Naive Bayes** algorithm.

## ⚙️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
* **Model:** Multinomial Naive Bayes (MultinomialNB)
* **Workflow:** Sklearn Pipeline

## 🧠 Approach & Workflow

1.  **Data Ingestion:** * Merged `True.csv` and `Fake.csv` datasets.
    * Combined `Title` and `Text` columns for better context.
2.  **Text Preprocessing:**
    * Removed punctuation and special characters.
    * **Stopword Removal:** Used NLTK to filter out common words (e.g., 'the', 'is') that add noise.
3.  **Vectorization (NLP Core):**
    * Converted text into numerical vectors using **Bag of Words (CountVectorizer)** followed by **TF-IDF Transformation** to weigh important words.
4.  **Model Training:** * Implemented **Multinomial Naive Bayes**, which is highly effective for text classification tasks.
    * Used `sklearn.pipeline` to streamline the workflow.
5.  **Evaluation:** * Achieved **97% Accuracy** on the test set.

## 📊 Model Performance
The model was evaluated using a Classification Report to check Precision and Recall, ensuring minimal false positives.

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **97%** |
| **Precision** | 0.96 |
| **Recall** | 0.98 |
| **F1-Score** | 0.97 |

## 📂 Dataset
The dataset consists of approx. 44,000 news articles categorized as:
* **True News:** Articles from legitimate sources.
* **Fake News:** Articles flagged as misinformation.
*(Source: Kaggle - Fake and Real News Dataset)*
