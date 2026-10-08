\# 🛡️ HateSense ML Dashboard



An interactive \*\*Hate Speech Detection and Machine Learning Dashboard\*\* built with Python and Streamlit.



HateSense allows users to upload a dataset, explore and clean text data, select features, train different machine learning models, evaluate their performance, and classify new text as \*\*Hate/Toxic\*\* or \*\*Non-Hate\*\*.



\## 🚀 Features



\- 📂 Upload CSV datasets

\- ⚡ Built-in demo dataset

\- 📊 Exploratory Data Analysis (EDA)

\- 🧹 Automatic text cleaning

\- 🔍 TF-IDF feature extraction

\- ✂️ Train/Test data splitting

\- 🤖 Multiple machine learning models

\- 📈 Accuracy, Precision, Recall and F1 Score

\- 🗂️ Confusion Matrix

\- 📋 Classification Report

\- ✍️ Single text prediction

\- 📋 Batch text prediction

\- 🌌 Interactive cyber-style ML dashboard



\## 🔄 ML Pipeline



The application follows an 8-step machine learning workflow:



1\. \*\*Data Input\*\*

2\. \*\*Exploratory Data Analysis\*\*

3\. \*\*Text Cleaning\*\*

4\. \*\*Feature Selection\*\*

5\. \*\*Train/Test Split\*\*

6\. \*\*Model Selection\*\*

7\. \*\*Model Training\*\*

8\. \*\*Metrics \& Live Prediction\*\*



\## 🤖 Machine Learning Models



HateSense supports the following classification models:



\- Logistic Regression

\- Linear SVM

\- SGD Classifier

\- Naive Bayes



\## 🧠 Text Processing



The text preprocessing pipeline includes:



\- Converting text to lowercase

\- Removing URLs

\- Removing user mentions

\- Removing special characters

\- Removing unnecessary spaces

\- TF-IDF vectorization

\- Configurable n-gram range

\- Configurable maximum TF-IDF features

\- Stop-word removal



\## 📊 Model Evaluation



The dashboard evaluates trained models using:



\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- Confusion Matrix

\- Classification Report



\## 🔍 Prediction



After training a model, users can:



\### Single Text Detection

Enter a single comment or text and classify it as:



\- ⚠️ \*\*HATE / TOXIC CONTENT\*\*

\- ✅ \*\*NON-HATE CONTENT\*\*



\### Batch Prediction

Enter multiple texts, one per line, and receive predictions for all inputs in a table.



\## 🛠️ Technologies Used



\- Python

\- Streamlit

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Regular Expressions (re)



\## 📂 Project Structure



```text

hate\_comment\_detector/

│

├── app.py

├── requirements.txt

└── README.mds

