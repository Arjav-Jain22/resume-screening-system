"""
Job Classification Module
Loads trained model and predicts job category from resume text
"""

import pickle
import numpy as np
import os

# Path to models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')


def load_classification_models():
    """
    Load all classification artifacts
    
    Returns:
    tuple : (model, vectorizer, label_encoder, categories, metadata)
    """
    try:
        with open(os.path.join(MODEL_DIR, 'job_classifier_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'categories.pkl'), 'rb') as f:
            categories = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'training_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        return model, vectorizer, label_encoder, categories, metadata
    
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return None, None, None, None, None


def predict_category(cleaned_resume_text, model, vectorizer, label_encoder, top_n=5):
    """
    Predict job category from cleaned resume text
    
    Parameters:
    cleaned_resume_text : str
        Preprocessed resume text
    model : trained classifier
    vectorizer : fitted TfidfVectorizer
    label_encoder : fitted LabelEncoder
    top_n : int
        Number of top predictions to return
    
    Returns:
    dict with prediction results
    """
    if not cleaned_resume_text or len(cleaned_resume_text.strip()) == 0:
        return {
            'predicted_category': 'Unknown',
            'confidence': 0.0,
            'top_predictions': [],
            'success': False,
            'error': 'Empty resume text'
        }
    
    try:
        # Vectorize
        features = vectorizer.transform([cleaned_resume_text])
        
        # Predict
        prediction = model.predict(features)[0]
        predicted_category = label_encoder.classes_[prediction]
        
        # Get confidence scores
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(features)[0]
            exp_scores = np.exp(decision - np.max(decision))
            probabilities = exp_scores / exp_scores.sum()
        else:
            probabilities = np.zeros(len(label_encoder.classes_))
            probabilities[prediction] = 1.0
        
        # Top N predictions
        top_indices = probabilities.argsort()[-top_n:][::-1]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                'category': label_encoder.classes_[idx],
                'confidence': float(probabilities[idx]) * 100
            })
        
        return {
            'predicted_category': predicted_category,
            'confidence': float(probabilities[prediction]) * 100,
            'top_predictions': top_predictions,
            'all_probabilities': {
                label_encoder.classes_[i]: float(probabilities[i]) * 100
                for i in range(len(label_encoder.classes_))
            },
            'success': True,
            'error': None
        }
    
    except Exception as e:
        return {
            'predicted_category': 'Error',
            'confidence': 0.0,
            'top_predictions': [],
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    model, vectorizer, le, cats, meta = load_classification_models()
    if model:
        print(f"   Classification models loaded!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Categories: {len(cats)}")
        print(f"   Best F1: {meta['f1_weighted']:.4f}")
    else:
        print("    Failed to load models")