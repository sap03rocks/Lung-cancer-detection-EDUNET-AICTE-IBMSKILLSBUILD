import streamlit as st
import numpy as np  # If using NumPy for data processing
import pickle

model=pickle.load(open('modellr.pkl','rb'))

# Feature names and types (replace with your actual features)
features = [
    ("gender", "binary"),
    ("smoking", "binary"),
    ("fingers", "binary"),
    ("anxiety", "binary"),
    ("peer pressure", "binary"),
    ("any disease?", "binary"),
    ("fatigue", "binary"),
    ("allergy", "binary"),
    ("wheezing", "binary"),
    ("alcohol consumption", "binary"),
    ("coughing", "binary"),
    ("breath", "binary"),
    ("swallowing problems", "binary"),
    ("pain", "binary"),
    ("age", "numerical"),
]

# Set up Streamlit app
st.title("Lung Cancer Prediction ML App")

# Collect input for binary features
binary_values = []
for name, _ in features:
    if name != "age":
        value = st.selectbox(name, ["Yes", "No"])
        binary_values.append(1 if value == "Yes" else 0)
    
# Collect age input
age = st.number_input("Age", min_value=0, max_value=120)

# Predict button action
if st.button("Predict"):
    # Combine binary and age values (adjust based on your data processing approach)
    input_data = np.array([binary_values + [age]])  # Assuming NumPy usage

    # Load model and make prediction
    # Replace with your model loading and prediction logic
    prediction = model.predict_proba(input_data)[0, 1]  # Access probability of class 1

    # Display results
    #st.success("The predicted probability of lung cancer is {:.4f}".format(prediction))

    # Risk assessment (optional)
    if prediction > 0.99:
        st.markdown("**High risk of lung cancer. Consulting a doctor is strongly recommended.**")
    else:
      st.markdown("**low risk.**")

