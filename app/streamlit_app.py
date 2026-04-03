import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.utils.preprocessor import preprocess_text
from app.utils.resume_parser import parse_resume
from app.utils.classifier import load_classification_models, predict_category
from app.utils.matcher import (
    load_matching_models, compute_match,
    find_top_matching_jds, get_score_color, get_verdict
)

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# ============================================
# LOAD MODELS (cached)
# ============================================
@st.cache_resource
def load_all_models():
    """Load all models once and cache them"""
    clf_model, clf_vectorizer, label_encoder, categories, clf_metadata = load_classification_models()
    match_vectorizer, jd_vectors, jd_reference, match_metadata = load_matching_models()

    return {
        'clf_model': clf_model,
        'clf_vectorizer': clf_vectorizer,
        'label_encoder': label_encoder,
        'categories': categories,
        'clf_metadata': clf_metadata,
        'match_vectorizer': match_vectorizer,
        'jd_vectors': jd_vectors,
        'jd_reference': jd_reference,
        'match_metadata': match_metadata
    }


# ============================================
# HELPER FUNCTIONS
# ============================================
def create_gauge_chart(score, title="Matching Score"):
    """Create a gauge chart for matching score"""
    color = get_score_color(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={'suffix': '%', 'font': {'size': 40}},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': 'white',
            'borderwidth': 2,
            'steps': [
                {'range': [0, 15], 'color': '#fadbd8'},
                {'range': [15, 25], 'color': '#fdebd0'},
                {'range': [25, 40], 'color': '#fef9e7'},
                {'range': [40, 60], 'color': '#d5f5e3'},
                {'range': [60, 100], 'color': '#abebc6'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=60, b=30)
    )

    return fig


def create_category_bar_chart(top_predictions):
    """Create horizontal bar chart for category predictions"""
    categories = [p['category'] for p in top_predictions]
    confidences = [p['confidence'] for p in top_predictions]

    colors = ['#3498db' if i == 0 else '#85c1e9' for i in range(len(categories))]

    fig = go.Figure(go.Bar(
        x=confidences,
        y=categories,
        orientation='h',
        marker_color=colors,
        text=[f'{c:.1f}%' for c in confidences],
        textposition='auto'
    ))

    fig.update_layout(
        title='Top Category Predictions',
        xaxis_title='Confidence (%)',
        yaxis_title='',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def create_keyword_comparison_chart(common, missing, extra):
    """Create keyword comparison visualization"""
    fig = go.Figure(data=[
        go.Bar(
            name='✅ Matching',
            x=['Matching Keywords'],
            y=[len(common)],
            marker_color='#2ecc71'
        ),
        go.Bar(
            name='❌ Missing from Resume',
            x=['Missing Keywords'],
            y=[len(missing)],
            marker_color='#e74c3c'
        ),
        go.Bar(
            name='➕ Extra in Resume',
            x=['Extra Keywords'],
            y=[len(extra)],
            marker_color='#3498db'
        )
    ])

    fig.update_layout(
        title='Keyword Analysis Summary',
        yaxis_title='Count',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        barmode='group'
    )

    return fig


def render_keywords_html(keywords, tag_class):
    """Render keywords as HTML tags"""
    if not keywords:
        return "<p style='color: gray;'>None found</p>"
    html = ""
    for kw in keywords:
        html += f'<span class="{tag_class}">{kw}</span> '
    return html


# ============================================
# SIDEBAR
# ============================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📄 Resume Screening System")
        st.markdown("---")

        st.markdown("### 🔧 Features")
        st.markdown("""
        - 📤 Upload Resume (PDF/DOCX/TXT)
        - 🏷️ Job Category Classification
        - 📊 JD Matching Score
        - 🔑 Keyword Analysis
        - 📋 Top Matching JDs
        """)

        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Upload your resume
        2. Paste a job description
        3. Click **Analyze**
        4. View results and insights
        """)

        st.markdown("---")
        st.markdown("### 📊 Scoring Guide")
        st.markdown("""
        - 🟢 **≥ 60%** → Excellent Match
        - 🟡 **≥ 40%** → Good Match
        - 🟠 **≥ 25%** → Moderate Match
        - 🔴 **≥ 15%** → Weak Match
        - ⚫ **< 15%** → No Match
        """)

        st.markdown("---")

        # Model info
        models = load_all_models()
        if models['clf_metadata']:
            st.markdown("### ⚙️ Model Info")
            st.markdown(f"""
            - **Classifier:** {models['clf_metadata'].get('best_model_name', 'N/A')}
            - **F1 Score:** {models['clf_metadata'].get('f1_weighted', 0):.3f}
            - **Categories:** {models['clf_metadata'].get('num_categories', 0)}
            - **JDs in DB:** {models['match_metadata'].get('total_jds', 0):,}
            """)


# ============================================
# MAIN PAGE - HEADER
# ============================================
def render_header():
    st.markdown("""
        <div class="main-header">
            <h1>📄 Resume Screening System</h1>
            <p>AI-Powered Job Classification & JD Matching</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================
# MAIN PAGE - INPUT SECTION
# ============================================
def render_input_section():
    """Render the input section and return resume text + JD text"""

    col1, col2 = st.columns(2)

    resume_text = None
    jd_text = None

    with col1:
        st.markdown("### 📤 Upload Resume")

        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Paste Text"],
            horizontal=True,
            key="resume_input_method"
        )

        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload your resume",
                type=['pdf', 'docx', 'txt'],
                help="Supported formats: PDF, DOCX, TXT"
            )

            if uploaded_file:
                text, file_type, error = parse_resume(uploaded_file)
                if error:
                    st.error(f"❌ {error}")
                else:
                    resume_text = text
                    st.success(f"✅ {file_type} file parsed successfully! ({len(text.split())} words)")
                    with st.expander("📝 View Extracted Text"):
                        st.text_area("Extracted Resume Text", text, height=200, disabled=True)

        else:
            resume_text = st.text_area(
                "Paste your resume text here:",
                height=250,
                placeholder="Paste your full resume text here..."
            )
            if resume_text:
                st.info(f"📝 {len(resume_text.split())} words entered")

    with col2:
        st.markdown("### 📋 Job Description")

        jd_text = st.text_area(
            "Paste the job description here:",
            height=300,
            placeholder="Paste the full job description including requirements, qualifications, and skills..."
        )

        if jd_text:
            st.info(f"📝 {len(jd_text.split())} words entered")

    return resume_text, jd_text


# ============================================
# RESULTS SECTION - CLASSIFICATION
# ============================================
def render_classification_results(classification_result):
    """Render job classification results"""

    st.markdown("## 🏷️ Job Classification")

    if not classification_result['success']:
        st.error(f"Classification failed: {classification_result['error']}")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        # Main prediction card
        predicted = classification_result['predicted_category']
        confidence = classification_result['confidence']

        st.markdown(f"""
            <div class="score-card score-excellent">
                <h3>Predicted Category</h3>
                <div style="font-size: 1.5rem; font-weight: bold; color: #1e8449;">
                    {predicted}
                </div>
                <div style="font-size: 1.1rem; color: #555; margin-top: 8px;">
                    Confidence: {confidence:.1f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # Top predictions chart
        fig = create_category_bar_chart(classification_result['top_predictions'])
        st.plotly_chart(fig, use_container_width=True)


# ============================================
# RESULTS SECTION - JD MATCHING
# ============================================
def render_matching_results(matching_result):
    """Render JD matching results"""

    st.markdown("## 📊 JD Matching Analysis")

    if not matching_result['success']:
        st.error("Matching computation failed.")
        return

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Gauge chart
        fig = create_gauge_chart(matching_result['matching_percentage'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Verdict card
        verdict = matching_result['verdict']
        recommendation = matching_result['recommendation']
        score = matching_result['matching_percentage']

        if score >= 60:
            card_class = "score-excellent"
        elif score >= 40:
            card_class = "score-good"
        elif score >= 25:
            card_class = "score-moderate"
        elif score >= 15:
            card_class = "score-weak"
        else:
            card_class = "score-none"

        st.markdown(f"""
            <div class="score-card {card_class}">
                <h3>Verdict</h3>
                <div style="font-size: 1.3rem; font-weight: bold;">
                    {verdict}
                </div>
                <div style="font-size: 0.9rem; color: #555; margin-top: 10px;">
                    {recommendation}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        # Keyword stats
        st.markdown(f"""
            <div class="metric-card">
                <h3>Keyword Match Rate</h3>
                <div class="value">{matching_result['keyword_match_rate']:.0f}%</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="metric-card" style="margin-top: 10px;">
                <h3>Common | Missing | Extra</h3>
                <div class="value" style="font-size: 1.3rem;">
                    {len(matching_result['common_keywords'])} |
                    {len(matching_result['missing_keywords'])} |
                    {len(matching_result['extra_keywords'])}
                </div>
            </div>
        """, unsafe_allow_html=True)


# ============================================
# RESULTS SECTION - KEYWORD ANALYSIS
# ============================================
def render_keyword_analysis(matching_result):
    """Render keyword analysis section"""

    st.markdown("## 🔑 Keyword Analysis")

    # Chart
    fig = create_keyword_comparison_chart(
        matching_result['common_keywords'],
        matching_result['missing_keywords'],
        matching_result['extra_keywords']
    )
    st.plotly_chart(fig, use_container_width=True)

    # Keyword tags
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ✅ Matching Keywords")
        st.markdown(
            render_keywords_html(
                matching_result['common_keywords'],
                'keyword-tag-match'
            ),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### ❌ Missing from Resume")
        st.markdown(
            render_keywords_html(
                matching_result['missing_keywords'],
                'keyword-tag-missing'
            ),
            unsafe_allow_html=True
        )

    with col3:
        st.markdown("### ➕ Extra in Resume")
        st.markdown(
            render_keywords_html(
                matching_result['extra_keywords'],
                'keyword-tag-extra'
            ),
            unsafe_allow_html=True
        )


# ============================================
# RESULTS SECTION - TOP MATCHING JDS
# ============================================
def render_top_jds(cleaned_resume, models):
    """Render top matching JDs from dataset"""

    st.markdown("## 📋 Top Matching Jobs from Database")

    top_matches = find_top_matching_jds(
        cleaned_resume,
        models['match_vectorizer'],
        models['jd_vectors'],
        models['jd_reference'],
        top_n=5
    )

    if not top_matches:
        st.warning("Could not find matching JDs.")
        return

    for match in top_matches:
        score = match['matching_score']
        color = match['score_color']

        with st.expander(
            f"#{match['rank']} | {match['category']} | Score: {score:.1f}% {match['verdict']}"
        ):
            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown(f"""
                    <div style="
                        background-color: {color}20;
                        border: 2px solid {color};
                        border-radius: 10px;
                        padding: 15px;
                        text-align: center;
                    ">
                        <div style="font-size: 2rem; font-weight: bold; color: {color};">
                            {score:.1f}%
                        </div>
                        <div style="color: #555;">Match Score</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"**Category:** {match['category']}")
                st.markdown(f"**Description Preview:**")
                st.text(match['description_preview'][:250] + "...")


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # Render sidebar
    render_sidebar()

    # Render header
    render_header()

    # Load models
    models = load_all_models()

    # Check if models loaded
    if models['clf_model'] is None:
        st.error("❌ Classification models could not be loaded. Check the models/ directory.")
        st.stop()

    if models['match_vectorizer'] is None:
        st.error("❌ Matching models could not be loaded. Check the models/ directory.")
        st.stop()

    # Render input section
    resume_text, jd_text = render_input_section()

    st.markdown("---")

    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_clicked = st.button(
            "🔍 Analyze Resume",
            type="primary",
            use_container_width=True
        )

    if analyze_clicked:
        if not resume_text or len(resume_text.strip()) < 50:
            st.error("❌ Please provide resume text (minimum 50 characters).")
            return

        # Progress bar
        progress = st.progress(0)
        status = st.empty()

        # Step 1: Preprocess resume
        status.text("⏳ Step 1/4: Preprocessing resume...")
        progress.progress(10)
        cleaned_resume = preprocess_text(resume_text)
        progress.progress(25)

        if len(cleaned_resume.strip()) == 0:
            st.error("❌ Resume text could not be processed. Please check the content.")
            return

        # Step 2: Classification
        status.text("⏳ Step 2/4: Classifying job category...")
        progress.progress(40)
        classification_result = predict_category(
            cleaned_resume,
            models['clf_model'],
            models['clf_vectorizer'],
            models['label_encoder'],
            top_n=5
        )
        progress.progress(55)

        # Step 3: JD Matching (if JD provided)
        matching_result = None
        if jd_text and len(jd_text.strip()) > 20:
            status.text("⏳ Step 3/4: Computing JD match score...")
            cleaned_jd = preprocess_text(jd_text)
            matching_result = compute_match(
                cleaned_resume,
                cleaned_jd,
                models['match_vectorizer']
            )
        progress.progress(75)

        # Step 4: Find top JDs
        status.text("⏳ Step 4/4: Finding top matching jobs...")
        progress.progress(90)
        time.sleep(0.3)
        progress.progress(100)
        status.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress.empty()
        status.empty()

        # ============================================
        # RENDER RESULTS
        # ============================================
        st.markdown("---")
        st.markdown("# 📊 Analysis Results")

        # Resume info
        st.markdown(f"""
            <div class="info-box">
                <strong>📝 Resume Stats:</strong>
                Original: {len(resume_text.split())} words →
                After cleaning: {len(cleaned_resume.split())} words
            </div>
        """, unsafe_allow_html=True)

        # Classification results
        render_classification_results(classification_result)

        st.markdown("---")

        # JD Matching results
        if matching_result:
            render_matching_results(matching_result)
            st.markdown("---")
            render_keyword_analysis(matching_result)
        else:
            st.info("💡 Paste a Job Description above to get matching score and keyword analysis.")

        st.markdown("---")

        # Top matching JDs from database
        render_top_jds(cleaned_resume, models)

        st.markdown("---")

        # Cleaned text viewer
        with st.expander("🔍 View Preprocessed Text"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Cleaned Resume:**")
                st.text_area("", cleaned_resume, height=200, disabled=True, key="clean_resume")
            with col2:
                if jd_text:
                    cleaned_jd = preprocess_text(jd_text)
                    st.markdown("**Cleaned JD:**")
                    st.text_area("", cleaned_jd, height=200, disabled=True, key="clean_jd")


if __name__ == "__main__":
    main()