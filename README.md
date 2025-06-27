# Aspect-Based Sentiment Classification for E-Commerce Reviews  
> Project conducted: December 2024 (was not uploaded at the time)

Building a machine learning pipeline to classify customer sentiments across multiple product aspects in the e-commerce domain.

---

## Introduction

This project aims to extract structured insights from unstructured customer reviews collected from Hasaki.vn. By leveraging NLP and machine learning, it automatically identifies six key product aspects:
- Store
- Service
- Packaging
- Price
- Quality
- Others

Each review is classified by both aspect and sentiment polarity (Positive/Negative/Neutral), supporting customer experience analysis at a granular level.

---

## Dataset

### Data Source  
Data was crawled using Selenium from product pages and review sections on Hasaki.vn.

### Labeling Process  
A hybrid labeling approach was used:
- Gemini API: for automated tagging of aspects and sentiments.
- Manual verification: to improve reliability.

Each review may be linked to multiple aspects, each with a corresponding sentiment label.

---

## Tools & Technologies

- Languages: Python  
- Libraries: pandas, numpy, scikit-learn, gensim, tensorflow, keras  
- Embedding: Word2Vec  
- Model persistence: joblib, pickle  
- Development: Google Colab (please update file paths if running locally)

---

## Project Workflow

### 1. Data Crawling  
- File: `crawl_comment.ipynb`  
- Crawls product IDs and corresponding customer reviews  
- Saves raw review data to `data/data_crawl.xlsx`

### 2. Labeling  
- File: `code_label_gemini_api.ipynb`  
- Uses Gemini API to assign:
  - Aspect tags: e.g., Service, Packaging...
  - Sentiment: Positive, Negative, Neutral  
- Outputs structured file `data/data_label.xlsx` with one-hot encoded aspect columns

### 3. Data Preprocessing  
- File: `data_preprocessing.ipynb`  
- Includes:
  - Lowercasing, punctuation/special character removal
  - Emoji replacement
  - Tokenization using Vietnamese rules
  - Stopword removal using a curated `.txt` list
- Output: cleaned review sequences in `data/data_preprocess.xlsx`

Example wordcloud:

![wordcloud](assets/wordcloud.png)

### 4. Embedding & Tokenization  
- Trains a custom Word2Vec model
- Tokenizes input sequences using Keras Tokenizer
- Produces:
  - `word2vec_sentiment.model` (embedding)
  - `tokenizer.pkl` (used for training neural networks)

### 5. Model Training  
- Folder `model_code/` contains 6 training notebooks (1 per aspect)
- Models trained:
  - Logistic Regression
  - SVM
  - Random Forest
  - Neural Network  
- Best-performing model selected per aspect  
- Trained `.joblib` models stored in `model_file/`

### 6. Evaluation  
- Performed with:
  - Accuracy
  - Precision / Recall / F1-score
  - Confusion Matrix
  - Classification Report
  - ROC Curve

Example for aspect_quality:

![Eva](assets/eva.jpg)
![Confusion Matrix](assets/cf.jpg)  
![Classify](assets/classify.jpg)
![ROC Curve](assets/ROC.jpg)

Summary all:

| Aspect     | Model            | Accuracy |
|------------|------------------|----------|
| Service    | Neural Network   | 92.3%    |
| Store      | Random Forest    | 90.1%    |
| Packaging  | Random Forest    | 89.72%   |
| Others     | Random Forest    | 88.23%   |
| Price      | Neural Network   | 85.4%    |

---

## Result
Result from running `main.ipynb` with a Vietnamese review input

![Result](assets/Result.jpg)

---

## How to Use

### Option 1: Full Pipeline  
1. Run the following in order (on Colab):
   - `crawl_comment.ipynb`
   - `code_label_gemini_api.ipynb`
   - `data_preprocessing.ipynb`
   - Any training notebook in `model_code/`
   - `main.ipynb`

### Option 2: Pre-trained Models  
1. Download:
   - `embedding_model_file/`  
   - `model_file/`  

2. Update paths in `main.ipynb`  
3. Input a Vietnamese review → returns aspect-level sentiment

---

## Learning Outcomes

- Applied aspect-based sentiment analysis to real-world e-commerce data  
- Trained and compared multiple classification models  
- Built a functional inference pipeline ready for deployment  
- Practiced crawling, labeling, NLP preprocessing, model evaluation, and modular code organization
