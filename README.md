# Wine Review Analysis

## Overview
This repository contains the source code and analysis for a series of machine learning models trained on a dataset of wine reviews. My objective is to predict sentiment based on the textual content of each review.

## Dataset
The dataset used is `winemag-data-130k-v2.csv`, containing 130,000 wine reviews. I preprocessed this data by cleaning and transforming the text and handling missing values.

## Models
Three machine learning models evaluated:
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Convolutional Neural Network (CNN)**

## Features
- **TF-IDF Vectorization** for Logistic Regression and SVM.
- **Tokenized sequences** for CNN.

## Code
The repository includes Python scripts for data cleaning, preprocessing, model training, and evaluation. Key libraries used include `pandas`, `nltk`, `sklearn`, `tensorflow`, `TextBlob`, and `matplotlib`.

## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
