<<<<<<< HEAD
# condition_detc
=======
# 💊 SVM Sentiment Classifier & Smart Drug Recommender

This project is a Flask-based web application that:
- Predicts the **sentiment** (positive/negative) of medical drug reviews using an SVM model.
- Recommends **alternative medicines** if the review sentiment is negative.
- Dynamically filters drugs based on the selected medical condition.

---

## 🚀 Features

✅ Sentiment classification using SVM  
✅ User selects from predefined **target conditions**  
✅ Drug options automatically update based on selected condition  
✅ Suggests 3 top-rated alternative medicines if sentiment is negative  

---

## 📁 Folder Structure

svm_sentiment_app/
│
├── app.py # Flask web application
├── train_model.py # Script to preprocess and train SVM model
├── model/
│ └── svm_sentiment_model.pkl # Trained model saved here
├── templates/
│ └── index.html # Frontend HTML template
├── drugsCom_raw.xlsx # Dataset (input reviews)
├── requirements.txt # All dependencies
└── README.md # You're here!

▶️ To Run Locally

---Go to the project directory

---Create and activate virtual environment

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate


---Install all dependencies
pip install -r requirements.txt


---Train the model
python train_model.py


---Run the Flask app
python app.py


---Open in browser
http://127.0.0.1:5000


>>>>>>> 094fbcf (readme.md)
