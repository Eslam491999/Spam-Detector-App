# Spam Detector App

A simple Streamlit app that uses machine learning to classify text messages as spam or not spam.

## Features
- Trains a Naive Bayes model on SMS spam data
- Allows users to input custom messages
- Displays prediction with confidence score
- Shows model evaluation metrics (confusion matrix, classification report)

## Requirements
See `requirements.txt`:
```text
streamlit
pandas
scikit-learn
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repo or download the files
2. Make sure `spam.csv` is in the same folder
3. Run the app:
```bash
streamlit run your_script_name.py
```

## Dataset
The app uses a sample of the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

## Example
- Input: "Congratulations! You've won a free ticket."
- Output: **Spam (0.98 confidence)**

## License
MIT
