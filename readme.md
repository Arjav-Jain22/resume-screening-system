# 📄 Resume Screening System

AI-powered Resume Screening System that performs **Job Classification** and **JD Matching** using NLP and Machine Learning.

## 🚀 Features

- **Resume Upload**: Support for PDF, DOCX, and TXT files
- **Job Classification**: Predicts job category using trained ML model (TF-IDF + Classifier)
- **JD Matching**: Computes cosine similarity between resume and job description
- **Keyword Analysis**: Identifies matching, missing, and extra keywords
- **Top Job Matches**: Finds best matching jobs from database of 19,000+ JDs

## 🛠️ Tech Stack

- **ML/NLP**: scikit-learn, NLTK, TF-IDF Vectorization
- **Frontend**: Streamlit, Plotly
- **Training**: Kaggle (GPU)
- **Deployment**: Streamlit Cloud / Local

## 📊 Model Performance

| Model | Accuracy | F1 (Weighted) | F1 (Macro) |
|-------|----------|---------------|------------|
| Best Model | XX.XX% | XX.XX% | XX.XX% |

## 📁 Project Structure

resume-screening-system/
├── models/ # Trained model artifacts
├── app/
│ ├── streamlit_app.py
│ └── utils/
│ ├── preprocessor.py
│ ├── resume_parser.py
│ ├── classifier.py
│ └── matcher.py
├── notebooks/ # Training notebooks
├── requirements.txt
└── README.md


## 🏃 How to Run

```bash
# Clone repo
git clone <your-repo-url>
cd resume-screening-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/streamlit_app.py

📈 Datasets Used

Resume Dataset by Snehaan Bhawal (Kaggle) - Job Classification
Job Postings Dataset by Akshat Jain (Kaggle) - JD Matching

📐 Methodology

Job Classification
Text Preprocessing → TF-IDF Vectorization → Multi-class Classification
25 job categories

JD Matching
Shared TF-IDF vocabulary (Resumes + JDs)
Cosine Similarity scoring
Keyword extraction and gap analysis

📝 Scoring Guide

Score	Verdict
≥ 60%	🟢 Excellent Match
≥ 40%	🟡 Good Match
≥ 25%	🟠 Moderate Match
≥ 15%	🔴 Weak Match
< 15%	⚫ No Match