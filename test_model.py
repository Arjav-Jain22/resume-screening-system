import pickle

# Load all artifacts
with open('models/job_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(f" Model loaded: {type(model).__name__}")

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print(f" Vectorizer loaded: vocab size = {len(vectorizer.vocabulary_)}")

with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
print(f" Label Encoder loaded: {len(encoder.classes_)} categories")

with open('models/categories.pkl', 'rb') as f:
    categories = pickle.load(f)
print(f" Categories: {categories}")

with open('models/training_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
print(f" Metadata: {metadata['best_model_name']} | "
      f"F1: {metadata['f1_weighted']:.4f}")

print("\n ALL MODELS LOADED SUCCESSFULLY!")
