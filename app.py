import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re
import string

# Page config
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="wide"
)

# Title and description
st.title("📧 Spam Mail Prediction")
st.markdown("### Detect spam emails using Machine Learning")

# Load and train model (in production, you'd load a pre-trained model)
@st.cache_resource
def load_model():
    # Sample training data (replace with your actual dataset)
    sample_data = {
        'text': [
            'Free money now! Click here to claim your prize!',
            'Meeting scheduled for tomorrow at 3 PM',
            'URGENT: Your account will be suspended! Act now!',
            'Thanks for the presentation today',
            'Win a free iPhone! Limited time offer!',
            'Please review the attached document',
            'Congratulations! You have won $1000000!',
            'Can we reschedule our call?',
            'CLICK HERE FOR AMAZING DEALS!!!',
            'The project deadline is next week'
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham
    }
    
    df = pd.DataFrame(sample_data)
    
    # Text preprocessing function
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    
    # Preprocess and train
    df['processed_text'] = df['text'].apply(preprocess_text)
    pipeline.fit(df['processed_text'], df['label'])
    
    return pipeline

# Load model
model = load_model()

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Email Text")
    email_text = st.text_area(
        "Paste your email content here:",
        height=200,
        placeholder="Enter the email text you want to analyze..."
    )
    
    if st.button("🔍 Analyze Email", type="primary"):
        if email_text.strip():
            # Preprocess text
            def preprocess_text(text):
                text = text.lower()
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            processed_text = preprocess_text(email_text)
            
            # Make prediction
            prediction = model.predict([processed_text])[0]
            probability = model.predict_proba([processed_text])[0]
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            if prediction == 1:
                st.error("🚨 **SPAM DETECTED**")
                confidence = probability[1] * 100
            else:
                st.success("✅ **LEGITIMATE EMAIL**")
                confidence = probability[0] * 100
            
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probability breakdown
            st.subheader("📈 Probability Breakdown")
            prob_df = pd.DataFrame({
                'Category': ['Legitimate', 'Spam'],
                'Probability': [probability[0], probability[1]]
            })
            st.bar_chart(prob_df.set_index('Category'))
            
        else:
            st.warning("Please enter some email text to analyze.")

with col2:
    st.subheader("ℹ️ About")
    st.info("""
    This spam detector uses:
    - **TF-IDF Vectorization** for text processing
    - **Naive Bayes Classifier** for prediction
    - **Natural Language Processing** techniques
    
    **How it works:**
    1. Text preprocessing (lowercase, remove special chars)
    2. Feature extraction using TF-IDF
    3. Classification using trained ML model
    """)
    
    st.subheader("🎯 Tips")
    st.markdown("""
    **Common spam indicators:**
    - Excessive use of CAPS
    - Multiple exclamation marks
    - Urgent language
    - Prize/money offers
    - Suspicious links
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Scikit-learn")