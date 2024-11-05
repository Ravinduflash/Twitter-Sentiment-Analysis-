# Sentiment Analysis Project

## Overview

This repository contains a comprehensive **Sentiment Analysis** project designed to classify text data into three sentiment categories: **positive**, **negative**, and **neutral**. Using a variety of machine learning models and natural language processing (NLP) techniques, this project leverages popular libraries such as **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, and **NLTK** for data manipulation, visualization, and analysis.

## Features

- **Data Preprocessing**: Cleans and prepares text data, including tokenization, stopword removal, and lemmatization.
- **Model Training**: Implements multiple machine learning algorithms, allowing comparison across models. Models included:
  - Logistic Regression
  - Naïve Bayes Classifier
  - Support Vector Classifier (SVC)
  - Decision Trees
  - Random Forest Classifier
- **Model Evaluation**: Assesses performance using key metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- **Visualization**: Generates visualizations, including **word clouds** for each sentiment category, to provide insights into the most common words associated with each sentiment.

## Installation

To get started with this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.x installed. Use the following command to install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

- **Python 3.x**
- **Jupyter Notebook**
- Required Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `nltk`
  - `scikit-learn`
  - `wordcloud`

## Usage

1. **Data Loading**: Load your dataset into the Jupyter Notebook. Ensure the dataset has a column for text data and one for sentiment labels.
2. **Preprocessing**: Run preprocessing steps to clean, tokenize, and lemmatize the text data.
3. **Model Training**: Choose and train a model from the available options. Adjust hyperparameters as needed.
4. **Evaluation**: Evaluate model performance using metrics such as accuracy, F1-score, and AUC-ROC.
5. **Visualization**: Generate word clouds and other visualizations to analyze sentiment distribution within the data.

### Example Code

The following example demonstrates how to load a dataset and generate a word cloud for negative sentiment words:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Load your dataset
df = pd.read_csv('data.csv')

# Generate a word cloud for negative sentiment
negative_df = df[df['sentiment'] == 'negative']

# Function to generate a word cloud
def generate_wordcloud(data, title):
    text = " ".join(review for review in data['text'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis("off")
    plt.show()

generate_wordcloud(negative_df, 'Negative Sentiment WordCloud')
```

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. **Fork the Repository**.
2. **Create a New Branch**:
   ```bash
   git checkout -b feature-branch
   ```
3. **Make Your Changes** and commit them:
   ```bash
   git commit -m 'Add new feature'
   ```
4. **Push to the Branch**:
   ```bash
   git push origin feature-branch
   ```
5. **Create a New Pull Request**.

## Acknowledgments

- The **Jupyter Development Team** for JupyterLab.
- Contributors of the libraries used in this project, such as **NLTK**, **scikit-learn**, and **Seaborn**.
- [Insert any other acknowledgments here].

## Contact

For questions or suggestions, feel free to reach out:

- **Your Name** - dulshanravindu505@gmail.com
- **GitHub**: [G.M.Ravindu Dulshan](https://github.com/Ravinduflash)
```
