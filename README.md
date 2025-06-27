# Aspect-Based Sentiment Classification for E-Commerce Reviews  
> *Project conducted: 2024 (was not uploaded at the time)*

Building a machine learning pipeline to classify customer sentiments across various product aspects in the e-commerce domain.

---

## Introduction

This project was developed with the aim of enhancing customer experience understanding in e-commerce platforms by leveraging NLP and machine learning. The model was trained on review data crawled from **Hasaki.vn**, a popular Vietnamese online beauty store.

Each customer comment is analyzed and classified based on six key aspects:  
**Store**, **Service**, **Packaging**, **Price**, **Quality**, and **Others**.  
This aspect-based sentiment analysis allows businesses to pinpoint specific areas of excellence and improvement.

---

## Dataset

### Data Source
Customer reviews were scraped using the **Selenium** library from Hasaki.vn. The dataset contains:
- Product IDs and metadata
- Review content in Vietnamese
- Timestamps and other metadata

### Labeling Process
Aspects and sentiments were annotated using a combination of:
- **Gemini API**
- **Manual labeling**

Labeled data was structured and exported for training via `data_label.xlsx`.

---

## Tools & Technologies

- **Languages**: Python  
- **Libraries**: Pandas, NumPy, scikit-learn, Gensim (Word2Vec), TensorFlow/Keras  
- **Crawling**: Selenium  
- **Model Management**: joblib, pickle  

---

## Project Workflow

1. **Data Collection**  
   - File: `crawl_comment.ipynb`  
   - Crawls review data using Selenium and saves to `data_crawl.xlsx`

2. **Labeling**  
   - File: `code_label_gemini_api.ipynb`  
   - Uses Gemini API and manual refinement to generate `data_label.xlsx`

3. **Preprocessing**  
   - File: `data_preprocessing.ipynb`  
   - Includes cleaning text, removing stopwords, replacing emojis, and lowercasing

4. **Feature Extraction (Word Embedding)**  
   - Trains Word2Vec model from scratch  
   - Saved in `embedding_model_file/word2vec_sentiment.model`  
   - Tokenizer saved as `tokenizer.pkl`

5. **Model Training**  
   - Separate models trained per aspect (see folder `model_code/`)
   - Algorithms used: Logistic Regression, SVM, Random Forest, Neural Network  
   - Final models saved in `model_file/` as `.joblib` files

6. **Evaluation**  
   - 5-Fold cross-validation performed for each aspect  
   - Accuracy results:
     - `Service`: Neural Network (92.3%)  
     - `Price`: Neural Network (85.4%)  
     - `Store`: Random Forest (90.1%)  
     - `Packaging`: Random Forest (89.72%)  
     - `Others`: Random Forest (88.23%)

---

## How to Use

### Option 1: Run Full Pipeline
To reproduce results from scratch:
1. Run files in this order:
   - `crawl_comment.ipynb`
   - `code_label_gemini_api.ipynb`
   - `data_preprocessing.ipynb`
   - One or more files in `model_code/`
   - `main.ipynb`

### Option 2: Use Pre-trained Models
To run directly using trained models:
1. Download all files from:
   - `embedding_model_file/`
   - `model_file/`

2. In `main.ipynb`, update the path to match your local directory for loading models.

3. Run `main.ipynb` and input a new review.  
   The model will return predicted **aspect(s)** and **sentiment**.

---

## Learning Outcomes

- Applied aspect-based sentiment analysis to real-world e-commerce data  
- Trained and compared multiple classification models  
- Built a functional inference pipeline ready for deployment  
- Practiced crawling, labeling, NLP preprocessing, model evaluation, and modular code organization

---

## File Structure

