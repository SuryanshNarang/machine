from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    clf, scaler, label_encoders = pickle.load(file)

def preprocess_data(data):
    for column in data.columns:
        if column in label_encoders:
            data[column] = label_encoders[column].transform(data[column])

    return data

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    form_df = pd.DataFrame(form_data, index=[0])
    processed_data = preprocess_data(form_df)

    processed_data_scaled = scaler.transform(processed_data)
    prediction = clf.predict(processed_data_scaled)

    target_prediction = label_encoders['Credit classification'].inverse_transform(prediction)

    return render_template("result.html", prediction=target_prediction[0])


if __name__ == "__main__":
    app.run(debug=True)