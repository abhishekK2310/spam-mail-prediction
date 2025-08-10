import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re

# Page config
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="wide"
)

# Title and description
st.title("📧 Spam Mail Prediction")
st.markdown("### Detect spam emails using Machine Learning")

# Load data, train model, and return the fitted model, vectorizer, and accuracy
@st.cache_resource
def load_model_and_vectorizer():
    # Load the dataset from the CSV file
    try:
        raw_mail_data = pd.read_csv('mail_data.csv')
    except FileNotFoundError:
        st.error("Error: 'mail_data.csv' not found. Please make sure the CSV file is in the same directory as the script.")
        return None, None, None

    # Replace null values with an empty string
    mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

    # Label spam as 0 and ham as 1
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

    # Separating the data as texts and label
    X = mail_data['Message']
    Y = mail_data['Category'].astype('int')

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Transform the text data to feature vectors
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)

    # Calculate accuracy on test data
    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

    return model, feature_extraction, accuracy_on_test_data

# Load the model, vectorizer, and accuracy
model, vectorizer, accuracy = load_model_and_vectorizer()

# Main interface
if model and vectorizer:  # Only run the interface if the model loaded successfully
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Enter Email Text")
        email_text = st.text_area(
            "Paste your email content here:",
            height=250,
            placeholder="Enter the email text you want to analyze..."
        )
        
        if st.button("🔍 Analyze Email", type="primary"):
            if email_text.strip():
                # Convert text to feature vectors
                input_data_features = vectorizer.transform([email_text])
                
                # Make prediction
                prediction = model.predict(input_data_features)[0]
                probability = model.predict_proba(input_data_features)[0]
                
                # Display results
                st.subheader("📊 Analysis Results")
                
                # 0 is Spam, 1 is Ham (as per your script)
                if prediction == 0:
                    st.error("🚨 **SPAM DETECTED**")
                    confidence = probability[0] * 100
                else:
                    st.success("✅ **LEGITIMATE EMAIL (HAM)**")
                    confidence = probability[1] * 100
                
                st.metric("Confidence", f"{confidence:.1f}%")
                
                # Probability breakdown
                st.subheader("📈 Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Category': ['Spam', 'Legitimate (Ham)'],
                    'Probability': [probability[0], probability[1]]
                })
                st.bar_chart(prob_df.set_index('Category'))
                
            else:
                st.warning("Please enter some email text to analyze.")

    with col2:
        st.subheader("ℹ️ About")
        st.info("""
        This spam detector uses the logic from your script:
        - **TF-IDF Vectorization** for text processing.
        - **Logistic Regression Classifier** for prediction.
        - **Train-Test Split** for model evaluation.
        """)
        
        # Display model accuracy
        st.subheader("🎯 Model Performance")
        st.write(f"The model was trained on the `mail_data.csv` dataset and achieved an accuracy of **{accuracy:.2%}** on the test set.")
        
        st.subheader("💡 Common Spam Indicators")
        st.markdown("""
        - Excessive use of **CAPS**.
        - Multiple **exclamation marks!!!**
        - **Urgent** or demanding language ("Act now!").
        - Unbelievable **prize/money offers**.
        - Suspicious **links** or requests for personal info.
        """)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️by Abhishek Kumar")
