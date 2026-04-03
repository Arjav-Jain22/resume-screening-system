"""Quick test to verify JD matching artifacts work"""
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load artifacts
with open('models/jd_matching_tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print(f" JD Vectorizer loaded: vocab size = {len(vectorizer.vocabulary_)}")

with open('models/jd_vectors.pkl', 'rb') as f:
    jd_vectors = pickle.load(f)
print(f" JD Vectors loaded: shape = {jd_vectors.shape}")

jd_ref = pd.read_csv('models/jd_reference_data.csv')
print(f" JD Reference loaded: {jd_ref.shape[0]} JDs")

with open('models/matching_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
print(f" Metadata loaded: {metadata['vocabulary_size']} vocab, {metadata['total_jds']} JDs")

# Quick matching test
test_text = "python developer machine learning tensorflow aws docker"
test_vector = vectorizer.transform([test_text])
scores = cosine_similarity(test_vector, jd_vectors)[0]
top_idx = scores.argsort()[-3:][::-1]

print(f"\nTop 3 JD matches for '{test_text}':")
for idx in top_idx:
    print(f"  Score: {scores[idx]*100:.1f}% | Category: {jd_ref.iloc[idx]['Category']}")

print("\n ALL JD MATCHING ARTIFACTS WORKING !")
