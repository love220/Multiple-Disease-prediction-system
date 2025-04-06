import os
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Cure",
                   layout="wide",
                   page_icon="👨‍🔬")

# Define the correct model directory
model_path = "C:/Users/Lenovo/Desktop/saved models"

# Function to load a model safely
def load_model(filename):
    file_path = os.path.join(model_path, filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as model_file:
            return pickle.load(model_file)
    else:
        st.error(f"Error: Model file '{filename}' not found in '{model_path}'")
        return None

# Load models
diabetes_model = load_model("diabetes_model.sav")
heart_disease_model = load_model("heart_disease_model.sav")
parkinsons_model = load_model("parkinsons_model.sav")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Function to generate and display multiple charts based on model output
def show_multiple_charts(prediction, title):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns

    x = np.linspace(0, 10, 100)

    # Line Plot
    axes[0, 0].plot(x, np.sin(x) * prediction, color='blue')
    axes[0, 0].set_title(f"Sine Wave (Scaled {title})")

    # Histogram
    axes[0, 1].hist(np.random.randn(1000) * prediction, bins=30, color='red')
    axes[0, 1].set_title(f"Histogram ({title})")

    # Scatter Plot
    axes[0, 2].scatter(np.random.rand(50) * prediction, np.random.rand(50), color='green')
    axes[0, 2].set_title(f"Scatter ({title})")

    # Bar Chart
    axes[1, 0].bar(["A", "B", "C"], [10, 20, 15], color='purple')
    axes[1, 0].set_title(f"Bar Chart ({title})")

    # Box Plot
    axes[1, 1].boxplot(np.random.randn(100) * prediction)
    axes[1, 1].set_title(f"Box Plot ({title})")

    # Hide empty subplot
    axes[1, 2].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result') and diabetes_model:
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]
            diab_prediction = diabetes_model.predict([user_input])[0]
            result_text = 'The person is diabetic' if diab_prediction == 1 else 'The person is not diabetic'
            st.success(result_text)
            show_multiple_charts(diab_prediction, "Diabetes")
        except ValueError:
            st.error("Please enter valid numerical values.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    cp = st.text_input('Chest Pain types')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholesterol in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    restecg = st.text_input('Resting Electrocardiographic results')
    thalach = st.text_input('Maximum Heart Rate achieved')
    exang = st.text_input('Exercise Induced Angina')
    oldpeak = st.text_input('ST depression induced by exercise')
    slope = st.text_input('Slope of the peak exercise ST segment')
    ca = st.text_input('Major vessels colored by fluoroscopy')
    thal = st.text_input('Thal')

    if st.button('Heart Disease Test Result') and heart_disease_model:
        try:
            user_input = list(map(float, [age, sex, cp, trestbps, chol, fbs,
                                          restecg, thalach, exang, oldpeak, slope, ca, thal]))
            heart_prediction = heart_disease_model.predict([user_input])[0]
            result_text = 'The person has heart disease' if heart_prediction == 1 else 'The person does not have heart disease'
            st.success(result_text)
            show_multiple_charts(heart_prediction, "Heart Disease")
        except ValueError:
            st.error("Please enter valid numerical values.")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    input_fields = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
                    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    user_inputs = [st.text_input(label) for label in input_fields]

    if st.button("Parkinson's Test Result") and parkinsons_model:
        try:
            user_input = list(map(float, user_inputs))
            parkinsons_prediction = parkinsons_model.predict([user_input])[0]
            result_text = "The person has Parkinson's disease" if parkinsons_prediction == 1 else "The person does not have Parkinson's disease"
            st.success(result_text)
            show_multiple_charts(parkinsons_prediction, "Parkinson's")
        except ValueError:
            st.error("Please enter valid numerical values.")
