import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

# Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict test to validate
y_pred = model.predict(X_test_vec)

# Streamlit UI
st.title("Spam Detector")
st.write("Classify your message as spam or not")

user_input = st.text_area("Enter your message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        probability = model.predict_proba(input_vec)[0][prediction]
        label = "Spam" if prediction == 1 else "Not Spam"
        st.success(f"Prediction: {label} ({probability:.2f} confidence)")

# Optional: Show model performance
with st.expander("Show model evaluation"):
    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))
