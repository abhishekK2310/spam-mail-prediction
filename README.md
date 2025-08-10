# 📧 Spam Mail Prediction

A machine learning-powered web application to detect spam emails using Streamlit, Scikit-learn, and NLP techniques.

## Features

- **Real-time spam detection** using Logistic Regression
- **Interactive web interface** built with Streamlit
- **Text preprocessing** with NLP techniques
- **Confidence scoring** for predictions
- **Visual probability breakdown**

## Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **NLP**: TF-IDF Vectorization
- **Deployment**: Render

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/abhishekK2310/spam-mail-prediction.git
cd spam-mail-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501`

## How It Works

1. **Text Preprocessing**: Converts text to lowercase, removes special characters
2. **Feature Extraction**: Uses TF-IDF vectorization to convert text to numerical features
3. **Classification**: Employs Naive Bayes classifier for spam detection
4. **Results**: Displays prediction with confidence score and probability breakdown

## Usage

1. Enter email text in the text area
2. Click "Analyze Email" button
3. View the prediction results and confidence score
4. Check the probability breakdown chart

## Model Performance

The model uses a Multinomial Naive Bayes classifier with TF-IDF features, which is effective for text classification tasks like spam detection.

## Deployment

This app is deployed on Render for easy access and scalability.

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

MIT License
