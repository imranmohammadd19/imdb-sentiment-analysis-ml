import joblib

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")

vectorizer = joblib.load("models/vectorizer.pkl")

while True:
    text = input("\nEnter a review (or type 'exit'): ")

    if text.lower() == "exit":
        break

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    if prediction == 1:
        print("Prediction: Positive ✅")
    else:
        print("Prediction: Negative ❌")
