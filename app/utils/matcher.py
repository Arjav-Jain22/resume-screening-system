"""
JD Matching Module
Computes cosine similarity between resume and job description
"""

import pickle
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# Path to models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')


def load_matching_models():
    """
    Load all JD matching artifacts
    
    Returns:
    tuple : (vectorizer, jd_vectors, jd_reference, metadata)
    """
    try:
        with open(os.path.join(MODEL_DIR, 'jd_matching_tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'jd_vectors.pkl'), 'rb') as f:
            jd_vectors = pickle.load(f)
        
        jd_reference = pd.read_csv(os.path.join(MODEL_DIR, 'jd_reference_data.csv'))
        
        with open(os.path.join(MODEL_DIR, 'matching_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        return vectorizer, jd_vectors, jd_reference, metadata
    
    except FileNotFoundError as e:
        print(f"Matching model file not found: {e}")
        return None, None, None, None


def get_verdict(score):
    """
    Get verdict based on matching score
    """
    if score >= 60:
        return "🟢 EXCELLENT MATCH", "Strong candidate. Resume aligns very well with the JD."
    elif score >= 40:
        return "🟡 GOOD MATCH", "Decent fit. Consider highlighting missing skills in your resume."
    elif score >= 25:
        return "🟠 MODERATE MATCH", "Partial match. Significant gaps exist. Tailor your resume to the JD."
    elif score >= 15:
        return "🔴 WEAK MATCH", "Poor alignment. Major skills/experience gaps identified."
    else:
        return "⚫ NO MATCH", "Resume does not align with this job description."


def get_score_color(score):
    """
    Get color for score visualization
    """
    if score >= 60:
        return "#2ecc71"
    elif score >= 40:
        return "#f1c40f"
    elif score >= 25:
        return "#e67e22"
    elif score >= 15:
        return "#e74c3c"
    else:
        return "#2c3e50"


def compute_match(cleaned_resume, cleaned_jd, vectorizer):
    """
    Compute matching score between resume and JD
    
    Parameters:
    cleaned_resume : str
        Preprocessed resume text
    cleaned_jd : str
        Preprocessed JD text
    vectorizer : fitted TfidfVectorizer
    
    Returns:
    dict with matching results
    """
    if not cleaned_resume or not cleaned_jd:
        return {
            'matching_score': 0.0,
            'matching_percentage': 0.0,
            'verdict': '⚫ NO MATCH',
            'recommendation': 'Could not process text.',
            'common_keywords': [],
            'missing_keywords': [],
            'resume_keywords': [],
            'jd_keywords': [],
            'keyword_match_rate': 0.0,
            'success': False
        }
    
    try:
        # Vectorize
        resume_vector = vectorizer.transform([cleaned_resume])
        jd_vector = vectorizer.transform([cleaned_jd])
        
        # Cosine similarity
        similarity = cosine_similarity(resume_vector, jd_vector)[0][0]
        score_pct = float(similarity * 100)
        
        # Extract keywords
        feature_names = vectorizer.get_feature_names_out()
        
        resume_tfidf = resume_vector.toarray()[0]
        jd_tfidf = jd_vector.toarray()[0]
        
        # Top keywords from each
        resume_top_idx = resume_tfidf.argsort()[-30:][::-1]
        jd_top_idx = jd_tfidf.argsort()[-30:][::-1]
        
        resume_keywords = set()
        for idx in resume_top_idx:
            if resume_tfidf[idx] > 0:
                resume_keywords.add(feature_names[idx])
        
        jd_keywords = set()
        for idx in jd_top_idx:
            if jd_tfidf[idx] > 0:
                jd_keywords.add(feature_names[idx])
        
        common_keywords = sorted(list(resume_keywords & jd_keywords))
        missing_keywords = sorted(list(jd_keywords - resume_keywords))
        extra_keywords = sorted(list(resume_keywords - jd_keywords))
        
        keyword_match_rate = (
            len(common_keywords) / max(len(jd_keywords), 1) * 100
        )
        
        verdict, recommendation = get_verdict(score_pct)
        
        return {
            'matching_score': float(similarity),
            'matching_percentage': score_pct,
            'verdict': verdict,
            'recommendation': recommendation,
            'score_color': get_score_color(score_pct),
            'common_keywords': common_keywords,
            'missing_keywords': missing_keywords,
            'extra_keywords': extra_keywords,
            'resume_keywords': sorted(list(resume_keywords)),
            'jd_keywords': sorted(list(jd_keywords)),
            'keyword_match_rate': keyword_match_rate,
            'success': True
        }
    
    except Exception as e:
        return {
            'matching_score': 0.0,
            'matching_percentage': 0.0,
            'verdict': '⚫ ERROR',
            'recommendation': f'Error computing match: {str(e)}',
            'common_keywords': [],
            'missing_keywords': [],
            'extra_keywords': [],
            'resume_keywords': [],
            'jd_keywords': [],
            'keyword_match_rate': 0.0,
            'success': False
        }


def find_top_matching_jds(cleaned_resume, vectorizer, jd_vectors, jd_reference, top_n=5):
    """
    Find top matching JDs from dataset for a given resume
    
    Parameters:
    cleaned_resume : str
        Preprocessed resume text
    vectorizer : fitted TfidfVectorizer
    jd_vectors : sparse matrix of pre-computed JD vectors
    jd_reference : DataFrame with JD info
    top_n : int
        Number of top matches to return
    
    Returns:
    list of dicts with top matching JDs
    """
    if not cleaned_resume:
        return []
    
    try:
        resume_vector = vectorizer.transform([cleaned_resume])
        similarities = cosine_similarity(resume_vector, jd_vectors)[0]
        
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(similarities[idx] * 100)
            verdict, _ = get_verdict(score)
            
            results.append({
                'rank': rank,
                'category': jd_reference.iloc[idx]['Category'],
                'description_preview': str(jd_reference.iloc[idx]['description_preview'])[:250],
                'matching_score': score,
                'verdict': verdict,
                'score_color': get_score_color(score)
            })
        
        return results
    
    except Exception as e:
        print(f"Error in batch matching: {e}")
        return []


if __name__ == "__main__":
    vectorizer, jd_vectors, jd_ref, metadata = load_matching_models()
    if vectorizer:
        print(f"   Matching models loaded!")
        print(f"   Vocabulary: {metadata['vocabulary_size']}")
        print(f"   JDs: {metadata['total_jds']}")
    else:
        print("   Failed to load matching models")