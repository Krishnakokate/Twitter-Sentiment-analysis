# Twitter sentiment analysis using NLP

## Overview

This project aims to perform sentiment analysis on text data using Natural Language Processing (NLP) techniques. The primary steps include data cleaning, visualization, feature extraction, model building, and hyperparameter tuning.

## Project Structure

1. **Importing Libraries:**
   - Import necessary Python libraries for data analysis, cleaning, visualization, and machine learning.

2. **Data Cleaning:**
   - Perform data cleaning tasks, including handling missing values, removing duplicates, and addressing any issues with the raw text data.

3. **Defining and Applying the Cleaner Function:**
   - Develop a function or set of functions for cleaning the text data and apply them to the entire dataset.

4. **Data Visualization:**
   - Visualize the distribution of sentiment in the dataset, explore word frequencies, and identify patterns in the text data.

5. **Plot for Cleaned Tweets:**
   - Generate plots and visualizations based on the cleaned tweets to gain insights into the distribution of sentiment.

6. **Data Visualization After Applying Stop Words:**
   - Visualize the impact of removing stop words on the dataset.

7. **Bag of Words Model (Feature Extraction):**
   - Implement a Bag of Words model to extract features from the text data.

8. **Logistic Model:**
   - Build a logistic regression model for sentiment analysis.

9. **Cross-Validation of Logistic Model:**
   - Evaluate the logistic regression model using cross-validation.

10. **Hyperparameter Tuning of Logistic Regression:**
    - Fine-tune hyperparameters of the logistic regression model for optimal performance.

11. **LR Model Without vs With Stop Words:**
    - Compare the performance of the logistic regression model with and without stop words.

12. **TF-IDF Vector BoW Model + Logistic Regression:**
    - Implement a TF-IDF Vectorizer combined with logistic regression for feature extraction and classification.

13. **Dimensionality Reduction:**
    - Explore techniques for dimensionality reduction to improve model efficiency.

14. **N-Gram Models:**
    - Experiment with N-Gram models to capture context and improve sentiment analysis.

15. **Word Embeddings:**
    - Explore and implement word embeddings techniques for better feature representation.


## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Krishnakokate/Twitter-Sentiment-analysis.git

2. Install required dependencies:
     ```bash
     cd nlp-sentiment-analysis
     pip install -r requirements.txt
3. Execute the provided notebooks or scripts for each step:
   ```bash
    sentiment_analysis_nlp.ipynb
