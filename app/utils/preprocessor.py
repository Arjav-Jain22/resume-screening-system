"""
Text Preprocessing Pipeline for Resume Screening System
Used for both Resume text and Job Description text
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Custom stop words to REMOVE (too common, no value)
CUSTOM_STOP_WORDS = {
    'resume', 'curriculum', 'vitae', 'cv', 'name', 'email',
    'phone', 'address', 'dear', 'sir', 'madam', 'sincerely',
    'regards', 'thank', 'thanks', 'looking', 'forward',
    'opportunity', 'position', 'company', 'organization',
    'please', 'find', 'attached', 'apply', 'application',
    'candidate', 'applicant', 'job', 'work', 'experience',
    'also', 'well', 'good', 'great', 'excellent',
    'would', 'could', 'should', 'may', 'might',
    'one', 'two', 'three', 'first', 'second',
    'new', 'year', 'years', 'month', 'months'
}

# Important skill keywords to Keep (never remove these)
IMPORTANT_KEYWORDS = {
    'python', 'java', 'javascript', 'react', 'angular', 'vue',
    'node', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
    'git', 'linux', 'html', 'css', 'api', 'rest', 'graphql',
    'machine', 'learning', 'deep', 'ai', 'nlp', 'cv',
    'tensorflow', 'pytorch', 'keras', 'scikit', 'pandas', 'numpy',
    'hadoop', 'spark', 'kafka', 'tableau', 'powerbi',
    'excel', 'word', 'powerpoint', 'photoshop', 'illustrator',
    'figma', 'sketch', 'wireframe', 'prototype',
    'agile', 'scrum', 'kanban', 'jira', 'confluence',
    'sales', 'marketing', 'seo', 'sem', 'analytics',
    'accounting', 'finance', 'audit', 'tax', 'compliance',
    'nursing', 'clinical', 'patient', 'medical', 'healthcare',
    'teaching', 'education', 'curriculum', 'training',
    'management', 'leadership', 'strategy', 'planning',
    'communication', 'teamwork', 'problem', 'solving',
    'data', 'analysis', 'database', 'visualization',
    'network', 'security', 'firewall', 'encryption',
    'android', 'ios', 'mobile', 'flutter', 'swift', 'kotlin',
    'devops', 'cicd', 'terraform', 'ansible',
    'blockchain', 'crypto', 'smart', 'contract',
    'cloud', 'saas', 'microservices', 'serverless',
    'testing', 'qa', 'automation', 'selenium', 'cypress',
    'ux', 'ui', 'design', 'frontend', 'backend', 'fullstack',
    'hr', 'recruitment', 'payroll', 'benefits',
    'legal', 'regulatory', 'intellectual', 'property',
    'supply', 'chain', 'logistics', 'inventory',
    'mechanical', 'electrical', 'civil', 'chemical',
    'automotive', 'aerospace', 'manufacturing',
    'consulting', 'advisory', 'research', 'development'
}


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)


def remove_urls(text):
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        r'[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.sub(' ', text)


def remove_emails(text):
    email_pattern = re.compile(r'\S+@\S+')
    return email_pattern.sub(' ', text)


def remove_phone_numbers(text):
    phone_pattern = re.compile(
        r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    )
    return phone_pattern.sub(' ', text)


def remove_special_characters(text):
    # Keep alphanumeric, spaces, and some useful chars
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
    # Handle special tech terms
    text = re.sub(r'c\+\+', 'cplusplus', text, flags=re.IGNORECASE)
    text = re.sub(r'c\#', 'csharp', text, flags=re.IGNORECASE)
    text = re.sub(r'\.net', 'dotnet', text, flags=re.IGNORECASE)
    text = re.sub(r'node\.js', 'nodejs', text, flags=re.IGNORECASE)
    text = re.sub(r'react\.js', 'reactjs', text, flags=re.IGNORECASE)
    text = re.sub(r'vue\.js', 'vuejs', text, flags=re.IGNORECASE)
    text = re.sub(r'next\.js', 'nextjs', text, flags=re.IGNORECASE)
    # Now remove remaining special chars
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text


def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_and_clean(text, remove_stopwords=True):
    tokens = word_tokenize(text.lower())

    cleaned_tokens = []
    for token in tokens:
        # Skip single characters (except important ones)
        if len(token) <= 1 and token not in {'r', 'c'}:
            continue

        # Keep important keywords regardless
        if token in IMPORTANT_KEYWORDS:
            cleaned_tokens.append(token)
            continue

        # Skip stop words
        if remove_stopwords and token in stop_words:
            continue

        # Skip custom stop words
        if token in CUSTOM_STOP_WORDS:
            continue

        # Skip pure numbers
        if token.isdigit():
            continue

        # Lemmatize
        token = lemmatizer.lemmatize(token, pos='v')
        token = lemmatizer.lemmatize(token, pos='n')

        if len(token) > 1:
            cleaned_tokens.append(token)

    return cleaned_tokens


def preprocess_text(text, return_tokens=False):
    """
    Complete preprocessing pipeline

    Parameters:
    text : str
        Raw text (resume or job description)
    return_tokens : bool
        If True, return list of tokens; else return cleaned string

    Returns:
    str or list : Cleaned text or list of tokens
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return [] if return_tokens else ""

    # Step 1: Remove HTML tags
    text = remove_html_tags(text)

    # Step 2: Remove URLs
    text = remove_urls(text)

    # Step 3: Remove emails
    text = remove_emails(text)

    # Step 4: Remove phone numbers
    text = remove_phone_numbers(text)

    # Step 5: Remove special characters (with tech term handling)
    text = remove_special_characters(text)

    # Step 6: Remove extra whitespace
    text = remove_extra_whitespace(text)

    # Step 7: Tokenize, remove stopwords, lemmatize
    tokens = tokenize_and_clean(text, remove_stopwords=True)

    if return_tokens:
        return tokens

    return ' '.join(tokens)


def preprocess_for_tfidf(text):
    """
    Lighter preprocessing for TF-IDF
    Keeps more words for better TF-IDF representation
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    text = text.lower()

    # Light stopword removal (only NLTK defaults)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 1 and not t.isdigit()
    ]

    return ' '.join(tokens)

# Testing

if __name__ == "__main__":
    # Test with sample resume text
    sample_resume = """
    <html><body>
    John Doe | john.doe@email.com | +1-234-567-8901
    https://linkedin.com/in/johndoe

    EXPERIENCE:
    Senior Python Developer at Google (2020-2023)
    - Developed machine learning models using TensorFlow and PyTorch
    - Built REST APIs using Node.js and React.js
    - Deployed on AWS using Docker & Kubernetes
    - Experience with C++ and .NET framework

    SKILLS: Python, Java, JavaScript, SQL, MongoDB, Git, Linux
    EDUCATION: B.Tech in Computer Science, MIT (2016-2020)
    </body></html>
    """

    sample_jd = """
    We are looking for a Senior Software Engineer with 5+ years of experience.

    Requirements:
    - Strong proficiency in Python and JavaScript
    - Experience with React.js, Node.js
    - Knowledge of AWS, Docker, Kubernetes
    - Experience with machine learning frameworks (TensorFlow/PyTorch)
    - Strong problem-solving and communication skills
    - Bachelor's degree in Computer Science or related field

    Benefits: Health insurance, 401k, Remote work
    Contact: hr@company.com | 555-123-4567
    """

    print("=" * 60)
    print("RESUME - Full Preprocessing:")
    print("=" * 60)
    print(preprocess_text(sample_resume))

    print("\n" + "=" * 60)
    print("RESUME - Token Output:")
    print("=" * 60)
    print(preprocess_text(sample_resume, return_tokens=True))

    print("\n" + "=" * 60)
    print("JD - Full Preprocessing:")
    print("=" * 60)
    print(preprocess_text(sample_jd))

    print("\n" + "=" * 60)
    print("JD - TF-IDF Preprocessing:")
    print("=" * 60)
    print(preprocess_for_tfidf(sample_jd))
