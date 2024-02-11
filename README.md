# Lung-cancer-detection-EDUNET-AICTE-IBMSKILLSBUILD
**Project Title:** Lung Cancer Detection with Machine Learning

**Author:** Saptarshi Mukherjee

**Date:** February 11, 2024

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E6smaTjMHv6MAKTu_eH90ASOj2AxNLVc?usp=sharing)

**Project Summary:**

This project explores the application of machine learning for lung cancer detection, utilizing scikit-learn's Logistic Regression, Support vector machines, Bernoulli Naive Bayes, Gaussian Naive Bayes, and Random Forest models. The project demonstrates deployemnt of an visual app on Google Colab using npx and Streamlit, creating a user-friendly web interface for model inference.

**Key Technologies:**

* Machine Learning: scikit-learn
* Model Development: Logistic Regression, SVM, BNB, GNB, Random Forest
* Jupyter Notebook: Google Colab
* Front-End Framework: Streamlit

**Key Findings:**
*the accuracy of the 5 models are as follows:
*The Accuracy of Logistic Regression is 87.5 %
*The Accuracy of Support Vector Machine is 85.71 %
*The Accuracy of Gaussian Naive Bayes is 91.07 %
*The Accuracy of Random Forest Classifier is 85.71 %
*The Accuracy of Bernoulli Naive Bayes is 91.07 %

**Project Structure:**

* **survey lung cancer.csv** the main dataset of the project
* **Lung-cancer-detection-EDUNET-AICTE-IBMSKILLSBUILD.ipynb** The main notebook of the project
* **app.py:** The Streamlit script for building the web interface.
* **requirements.txt:** Lists all Python dependencies required for the project.
* **README.md:** This file (you're reading it now!).

**Running the Project:**

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Open notebooks/main.ipynb in Jupyter Notebook:**
   - Execute code cells to develop, train, and evaluate models.
   - Adjust hyperparameters as needed.
3. **Run app.py:**
   ```bash
   streamlit run app.py
   ```
   - Access the Streamlit web interface in your browser to predict lung cancer risk.
