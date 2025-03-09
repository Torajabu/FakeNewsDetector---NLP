# FakeNewsDetector---NLP
The project leverages Python for its implementation, combining NLP techniques (NLTK, TfidfVectorizer, HashingVectorizer) with machine learning (Multinomial Naive Bayes) to classify news articles as real or fake. Pandas is used for data handling, Matplotlib for visualization, and Scikit-learn for model training, evaluation, and feature extraction. 
DATASET USED - https://www.kaggle.com/c/fake-news/data#
# Fake News Detection

A machine learning model to identify fake news articles using Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier.

## Overview

This project uses NLP and machine learning to classify news articles as either "FAKE" or "REAL". It implements TF-IDF vectorization and achieves high accuracy in detecting fake news.

![Code Execution Screenshot](https://github.com/Torajabu/FakeNewsDetector---NLP/blob/main/Screenshot%202025-03-09%20131533.png)

## Features

- Text preprocessing including stemming and stop-word removal
- TF-IDF vectorization with n-gram features
- Multinomial Naive Bayes classification
- Performance evaluation with accuracy metrics and confusion matrices
- Test dataset prediction and output generation

## Requirements

```
pandas
scikit-learn
numpy
matplotlib
nltk
```

## Dataset

The model uses two datasets:
- `train.csv`: Contains labeled news articles with their classification (FAKE or REAL)
- `test.csv`: Contains news articles that need to be classified

## File Structure

```
C:/CODING/WORKING/
├── train.csv       # Training dataset
├── test.csv        # Test dataset
├── output.csv      # Generated predictions
└── confusion_matrix.png  # Visualization of model performance
```

## Usage

1. Ensure all required packages are installed:
   ```
   pip install pandas scikit-learn numpy matplotlib nltk
   ```

2. Download required NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

3. Update the file paths in the script to match your directory structure:
   ```python
   train_path = "path/to/train.csv"
   test_path = "path/to/test.csv"
   output_path = "path/to/output.csv"
   ```

4. Run the script:
   ```
   python fake_news_detection.py
   ```

## How It Works

### Data Preprocessing
- Removes missing values from the dataset
- Cleans text by removing non-alphabetic characters
- Converts text to lowercase
- Removes English stop words
- Applies stemming to reduce words to their root form

### Feature Extraction
- Uses TF-IDF Vectorization to convert text into numerical features
- Implements n-grams (1 to 3) to capture phrases
- Limits to 5000 features to manage dimensionality

### Model Training and Validation
- Splits the training data into training (80%) and validation (20%) sets
- Trains a Multinomial Naive Bayes classifier
- Evaluates performance on the validation set
- Retrains on the full training dataset for final model

### Prediction
- Processes the test dataset using the same preprocessing steps
- Makes predictions using the trained model
- Saves the predictions to an output CSV file

## Results

The model achieves high accuracy on the validation set and generates predictions for the test dataset. The output is saved in [output.csv](https://github.com/Torajabu/FakeNewsDetector---NLP/blob/main/output.csv), which contains the article IDs and their predicted labels.

A confusion matrix visualization is generated to show the model's performance:
- True positives (correctly identified real news)
- True negatives (correctly identified fake news)
- False positives (fake news classified as real)
- False negatives (real news classified as fake)

## Future Improvements

- Experiment with different classifiers (SVM, Random Forest, etc.)
- Implement cross-validation for more robust evaluation
- Add feature selection techniques
- Try deep learning approaches like LSTM or BERT
- Create a web interface for real-time classification

## Repository

The project is available at [https://github.com/Torajabu/FakeNewsDetector---NLP](https://github.com/Torajabu/FakeNewsDetector---NLP)

## Author

Torajabu
