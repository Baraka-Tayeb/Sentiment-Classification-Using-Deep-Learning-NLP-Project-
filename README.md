# Sentiment-Classification-Using-Deep-Learning-NLP-Project-
This repository contains an end-to-end Deep Learning–based Sentiment Classification System built to automatically identify whether customer reviews express Negative, Neutral, or Positive sentiment.

Using a real-world Women’s E-Commerce Review dataset, the project demonstrates strong capabilities in text preprocessing, feature engineering, deep learning modeling, and model evaluation for sentiment analysis.

This system can be integrated into applications such as:
chatbots, product-review analytics, customer-insight dashboards, voice-of-customer systems, and SaaS feedback tools.

# 🎯 Objective

To build an intelligent sentiment-classification engine that:

Preprocesses and cleans real-world customer review text

Converts raw text into meaningful numerical features

Trains multiple machine learning and deep learning models

Classifies reviews into Negative, Neutral, or Positive

Evaluates model performance using accuracy, F1-score, confusion matrices, and ROC curves

This project showcases a full NLP workflow and practical experience in designing production-ready deep learning components.

# 🧠 Key Capabilities

Advanced Text Cleaning: Handles emojis, contractions, elongations, URLs, and noise

Tokenization & Lemmatization: spaCy-based linguistic normalization

TF-IDF Feature Engineering: N-gram features for textual representation

Deep Learning Models: Experimentation with logistic regression, SVM, and XGBoost

Sentiment Mapping: Converts ratings → Neg/Neutral/Pos labels

Evaluation & Visualization: Confusion matrices, ROC curves, F1 scores

Interactive Interface: Gradio app for real-time predictions

# 🏗️ Technical Approach

The system follows a structured NLP + Deep Learning pipeline:

1. Data Preparation

Load and inspect the Women’s Clothing E-Commerce Reviews dataset

Handle missing values & duplicates

Map numerical ratings into sentiment classes

Perform exploratory data analysis (EDA) including:

Rating distribution

Review length analysis

Word clouds for each sentiment group

2. Text Preprocessing

Includes multiple layers of NLP cleaning:

URL removal

Contraction expansion

Emoji → text conversion

Normalization of elongated words

Tokenization with spaCy

Lemmatization

Custom stopword handling (preserving sentiment carriers like not, never, very)

Final output: a clean text column ready for vectorization.

3. Feature Engineering

TF-IDF vectorization with:

1–2 gram representation

5,000 max features

Accent stripping & sublinear term frequency

Produces sparse matrices fed into ML and DL models

4. Model Training

Multiple models were explored:

✔ Logistic Regression

Baseline classifier with strong linear performance.

✔ Linear SVM

High-margin classifier, excellent for sparse text features.

✔ XGBoost

Tree-based gradient boosting for nonlinear pattern detection.

Each model was tuned using RandomizedSearchCV and evaluated on:

Accuracy

Macro F1-score

Classification report

Confusion matrix

Multiclass ROC curves

5. Evaluation

# Comprehensive evaluation includes:

Per-class precision, recall, F1-score

Macro-averaged performance metrics

Confusion matrices for error interpretation

ROC curves and AUC values for all three sentiment classes

Insights into model performance and misclassification patterns

6. Deployment Interface

A Gradio web app allows real-time predictions:

User enters review text

# Model outputs:

Negative 😠

Neutral 😐

Positive 😊

This makes the system practical for demo, integration, and end-user interaction.

# 🛠️ Technologies Used

Python

spaCy (tokenization & lemmatization)

NLTK / TextBlob (linguistic utilities)

NumPy, Pandas (data wrangling)

Matplotlib, Seaborn (visualization)

Scikit-Learn (ML models & tuning)

XGBoost (advanced boosting)

TF-IDF Vectorizer

Gradio (deployment interface)

Jupyter Notebook (pipeline development)
