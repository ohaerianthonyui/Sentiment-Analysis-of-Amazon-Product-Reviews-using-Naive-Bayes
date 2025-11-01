# Sentiment-Analysis-of-Amazon-Product-Reviews-using-Naive-Bayes




##  **Project Title:**

### **Sentiment Analysis of Amazon Product Reviews using Naive Bayes**

---

## **1. Introduction**

Sentiment analysis is a key application of Natural Language Processing (NLP) that aims to determine the emotional tone behind textual data. It’s widely used in industries such as marketing, customer service, and social media monitoring to understand public opinion, customer satisfaction, and brand perception.

This project focuses on building a **machine learning-based sentiment analysis model** using **Naive Bayes classification** to automatically predict the sentiment expressed in text reviews. The model categorizes each review into one of three sentiment classes:

* **0 → Negative**
* **1 → Neutral**
* **2 → Positive**

By analyzing large volumes of text data, the model can help stores and businesses gain insights into customer attitudes and improve decision-making.

---

##  **2. Objectives**

The main goals of this project are:

1. To preprocess and vectorize raw text data for machine learning.
2. To train a **Naive Bayes classifier** using textual features.
3. To evaluate the model using key metrics such as accuracy, precision, recall, and F1-score.
4. To visualize model performance through a confusion matrix.
5. To test the model with new, unseen review examples.

---

##  **3. Dataset Description**

The dataset used in this project consists of two columns:

| Column Name     | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| **review_text** | The text content of a review (string)                        |
| **label**       | Sentiment category (0 = Negative, 1 = Neutral, 2 = Positive) |

Example data:

| review_text                           | label |
| ------------------------------------- | ----- |
| “I loved this movie, it was amazing!” | 2     |
| “It was okay, not great but not bad.” | 1     |
| “Terrible film, waste of time.”       | 0     |

---

##  **4. Methodology**

### 4.1 Data Preprocessing

* Convert text to lowercase
* Remove stopwords and punctuation (using `CountVectorizer`’s preprocessing)
* Convert text into numerical feature vectors using **n-grams (unigrams + bigrams)**

### 4.2 Feature Extraction

Used **CountVectorizer** from scikit-learn:

```python
cv = CountVectorizer(ngram_range=(1,2), stop_words='english')
X = cv.fit_transform(corpus)
```

### 4.3 Model Selection

Trained a **Multinomial Naive Bayes** model:

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
```

### 4.4 Evaluation Metrics

The model was evaluated using:

* **Accuracy**
* **Precision, Recall, F1-score**
* **Confusion Matrix**

### 4.5 Prediction

The trained model predicts sentiment for new text samples.

---

##  **5. Results**

| Metric             | Score |
| ------------------ | ----- |
| **Accuracy**       | ~75%  |
| **Macro F1-score** | ~0.75 |

The confusion matrix showed the model performed best on **positive reviews**, while most misclassifications occurred between **neutral and negative** sentiments — a common challenge in sentiment analysis.

Example predictions:

```
"This movie was absolutely wonderful!" → Positive (2)
"It was okay, nothing special." → Neutral (1)
"I really hated this film, it was awful." → Negative (0)
```

---

##  **6. Conclusion**

This project successfully demonstrates how **Naive Bayes**, a simple yet powerful probabilistic algorithm, can effectively classify sentiment in text data. With minimal preprocessing and a bag-of-words representation, the model achieved a solid 75% accuracy.

Future improvements may include:

* Using **TF-IDF** features for better word weighting
* Trying **Logistic Regression** or **SVM** models
* Leveraging **pretrained transformer models (e.g., BERT)** for deeper semantic understanding

---

##  **7. Tools & Libraries**

| Library                  | Purpose                                            |
| ------------------------ | -------------------------------------------------- |
| **Python**               | Core programming language                          |
| **pandas**               | Data manipulation                                  |
| **scikit-learn**         | Model training, evaluation, and feature extraction |
| **NLTK**                 | Tokenization and preprocessing                     |
| **matplotlib / seaborn** | Data visualization                                 |

---

##  **8. References**

* NLTK Documentation: [https://www.nltk.org](https://www.nltk.org)
* Scikit-learn User Guide: [https://scikit-learn.org](https://scikit-learn.org)
* “Naive Bayes Classifier for Text Classification” — Research Papers and Tutorials

---
