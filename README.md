🚢 Titanic Survival Prediction using Random Forest & Streamlit

This project demonstrates an end-to-end Machine Learning pipeline using a Random Forest classification model, deployed as an interactive web application using Streamlit.

The application predicts whether a passenger would survive the Titanic disaster based on personal and travel details.

📌 Project Highlights

End-to-end ML workflow (data → model → deployment)

Random Forest Classification

Clean modular code structure

Interactive Streamlit web application

Ready for GitHub & Streamlit Cloud deployment

📂 Project Structure
rf_streamlit_classifier/
│
├── rf_venv/                    # Virtual environment (ignored in GitHub)
│
├── data/
│   └── Titanic-Dataset.csv     # Dataset
│
├── models/
│   └── rf_model.pkl            # Trained Random Forest model
│
├── src/
│   ├── preprocess.py           # Data preprocessing
│   ├── train_model.py          # Model training
│   └── evaluate.py             # Model evaluation
│
├── app.py                      # Streamlit application
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation

📊 Dataset Information

Dataset: Titanic Dataset

Target Variable: Survived

Features Used:

Passenger Class

Sex

Age

Siblings/Spouses

Parents/Children

Fare

Embarked Port

⚙️ Technologies Used

Python 3.10+

Pandas & NumPy

Scikit-learn

Random Forest Classifier

Streamlit

Joblib

🔧 Setup Instructions (Local)
1️⃣ Clone the repository
git clone https://github.com/yourusername/rf_streamlit_classifier.git
cd rf_streamlit_classifier

2️⃣ Create and activate virtual environment
python -m venv rf_venv
rf_venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🧠 Model Training

Train the Random Forest model:

python src/train_model.py


The trained model will be saved in the models/ directory.

📈 Model Evaluation

Evaluate model performance:

python src/evaluate.py


Metrics displayed:

Accuracy

Precision

Recall

F1-Score

🌐 Run Streamlit Application
streamlit run app.py


Open your browser and navigate to:

http://localhost:8501