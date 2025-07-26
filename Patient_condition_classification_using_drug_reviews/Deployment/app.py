from flask import Flask, request, render_template
import joblib
import pandas as pd
from collections import defaultdict

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load("model/svm_sentiment_model.pkl")

# Load and filter dataset
df = pd.read_excel("drugsCom_raw.xlsx")

# Only keep target conditions
target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
df = df[df['condition'].isin(target_conditions)]

# Build mapping: condition ➜ [list of drugs]
condition_drug_map = defaultdict(list)
for condition in target_conditions:
    drugs = df[df['condition'] == condition]['drugName'].unique().tolist()
    condition_drug_map[condition] = sorted(set(drugs))

# Function to suggest alternatives if review is negative
def suggest_alternative_medicine(condition, current_drug):
    filtered = df[(df['condition'] == condition) & 
                  (df['drugName'] != current_drug)]

    # Top 3 drugs by average rating
    top_alternatives = (
        filtered.groupby('drugName')['rating']
        .mean()
        .sort_values(ascending=False)
    )

    return top_alternatives.head(3).index.tolist()

# Home route
@app.route('/')
def index():
    return render_template("index.html", condition_drug_map=condition_drug_map)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    condition = request.form['condition']
    drug = request.form['drug']
    review = request.form['review']

    # Predict sentiment
    prediction = model.predict([review])[0]

    # Recommend alternatives if sentiment is negative
    alternatives = []
    if prediction == 'negative':
        alternatives = suggest_alternative_medicine(condition, drug)

    return render_template("index.html",
                           condition=condition,
                           drug=drug,
                           review=review,
                           prediction=prediction,
                           alternatives=alternatives,
                           condition_drug_map=condition_drug_map)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
