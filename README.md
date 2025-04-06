## 🩺 Multiple Disease Prediction System – ML & Streamlit Deployment

### 📌 Project Description:

The **Multiple Disease Prediction System** is a machine learning-based web application designed to predict the likelihood of three major chronic illnesses: **Diabetes**, **Heart Disease**, and **Parkinson’s Disease**. This tool empowers users—especially healthcare professionals and patients—with early diagnostic insights based on input symptoms and clinical parameters.

The system leverages three separate machine learning models trained on publicly available datasets, each specialized for a specific disease:

1. **Diabetes Prediction Model**  
   - Algorithm: Logistic Regression (or any used)  
   - Input features: Glucose level, BMI, Age, Blood Pressure, etc.

2. **Heart Disease Prediction Model**  
   - Algorithm: Random Forest Classifier (or any used)  
   - Input features: Chest pain type, cholesterol, resting ECG, maximum heart rate, etc.

3. **Parkinson's Disease Prediction Model**  
   - Algorithm: Support Vector Machine (SVM)  
   - Input features: MDVP:Fo(Hz), Jitter, Shimmer, HNR, RPDE, DFA, etc.

### 🧠 Machine Learning Workflow:
- Data Preprocessing and Cleaning
- Feature Selection
- Model Training and Evaluation
- Model Serialization using **Pickle**
- Deployment using **Streamlit** for a clean and interactive web interface

### 🌐 Deployment:
The application has been deployed using **Streamlit**, allowing real-time predictions via a user-friendly web interface. Users can choose the disease they want to test for, input relevant health data, and receive a prediction immediately.

### 📁 Folder Structure:
```
C:\ml_deployment
│
├── app.py                  # Streamlit web application
├── diabetes_model.pkl      # Trained diabetes prediction model
├── heart_model.pkl         # Trained heart disease model
├── parkinsons_model.pkl    # Trained Parkinson's model
├── requirements.txt        # Python dependencies
└── assets/ or helper files
```

### 🔥 Key Features:
- Predicts 3 diseases in one unified interface
- Accurate and fast real-time prediction
- Clean UI powered by Streamlit
- Easy to use for both patients and doctors

### 🛠️ Technologies Used:
- Python
- Machine Learning (Scikit-learn, Pandas, NumPy)
- Streamlit for deployment
- Pickle for model serialization
