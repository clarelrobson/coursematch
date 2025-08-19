import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configs
st.set_page_config(
    page_title="Course Match",
    page_icon="https://raw.githubusercontent.com/clarelrobson/credit-comparison-site/main/C-removebg-preview.png",
    layout="wide",
)

# --- Website styling ---
st.markdown("""
<style>
h1 { color: #1c3144; }
h2, h3 { color: #1c3144; }
body, div, p, li { color: #3a506b; }
.stTextArea textarea, .stTextInput input {
    color: #3a506b !important;
    background-color: #f0f4f8 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Cached SentenceTransformer model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2', device=device)

model = load_model()

# --- Cached CSV loader ---
@st.cache_data
def load_courses(university):
    university_files = {
        "Pennsylvania State University": "https://raw.githubusercontent.com/clarelrobson/coursematch/refs/heads/main/psu_courses.csv",
        "Temple University": "https://raw.githubusercontent.com/clarelrobson/coursematch/refs/heads/main/temple_courses.csv",
        "West Chester University of PA": "https://raw.githubusercontent.com/clarelrobson/coursematch/refs/heads/main/wcu_courses.csv"
    }
    df = pd.read_csv(university_files[university])
    required_columns = ['Subject', 'Course Title', 'Description']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"{university} courses CSV must contain: {', '.join(required_columns)}")
    return df

# --- Cached TF-IDF vectorizer ---
@st.cache_resource
def fit_subject_vectorizer(subjects):
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
    subject_vectors = vectorizer.fit_transform(subjects)
    return vectorizer, subject_vectors

def identify_relevant_subjects(sending_course_desc, subjects, vectorizer=None, subject_vectors=None):
    if vectorizer is None or subject_vectors is None:
        vectorizer, subject_vectors = fit_subject_vectorizer(subjects)
    course_vector = vectorizer.transform([sending_course_desc])
    similarities = cosine_similarity(course_vector, subject_vectors)[0]
    relevant = [subjects[i] for i in range(len(subjects)) if similarities[i] > 0.2]
    return relevant if relevant else subjects, vectorizer, subject_vectors

# --- Cached course embeddings ---
@st.cache_resource
def get_course_embeddings(courses_dict):
    return model.encode(list(courses_dict.values()), convert_to_tensor=True, device=device, batch_size=32)

# --- Compare course descriptions ---
def compare_courses_batch(sending_course_desc, filtered_courses, cached_embeddings=None):
    sending_vec = model.encode(sending_course_desc, convert_to_tensor=True, device=device)
    receiving_vecs = cached_embeddings if cached_embeddings is not None else model.encode(list(filtered_courses.values()), convert_to_tensor=True, device=device, batch_size=32)
    scores = util.pytorch_cos_sim(sending_vec, receiving_vecs)
    results = {title: scores[0][i].item() for i, title in enumerate(filtered_courses.keys())}
    return sorted(results.items(), key=lambda item: item[1], reverse=True)[:10]

# --- Color coding for similarity scores ---
def get_color(score):
    if score >= 0.8: return "#d0f0e9"
    elif score >= 0.6: return "#eaf9d6"
    elif score >= 0.4: return "#fff9e6"
    elif score >= 0.2: return "#ffe6cc"
    else: return "#ffd6cc"

# --- Streamlit Interface ---
def main():
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("""
            <h1 style="display: flex; align-items: center;">
            <img src="https://raw.githubusercontent.com/clarelrobson/credit-comparison-site/main/C-removebg-preview.png" 
            alt="logo" style="width: 40px; margin-right: 15px;"/> 
            Course Match
            </h1>
        """, unsafe_allow_html=True)

        # --- Intro section ---
        st.markdown("""
        This tool helps you determine how a course at one university (the **sending university**) compares to courses offered at another university (the **receiving university**). 

        - **Sending University**: The institution where the course you want to evaluate is offered. Enter the description of this course in the input box.
        - **Receiving University**: The institution where you want to see comparable courses. Select this university from the dropdown menu.

        By analyzing course descriptions using advanced [Natural Language Processing (NLP) techniques](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2), this tool identifies the top 10 most similar courses from the receiving university. Each result is scored to reflect how closely the course descriptions match.
        """)

        sending_course_desc = st.text_area("Enter the description for the sending university course")
        university = st.selectbox("Select the receiving university", ["Select...", "Pennsylvania State University", "Temple University", "West Chester University of PA"])

        # --- Similarity rating ---
        st.markdown("""
        <h3>Similarity Rating Explanation</h3>
        <div style="background-color:#d0f0e9; padding:5px; margin-bottom:5px;">
            <strong>0.8 - 1.0</strong>: Very High Similarity – Descriptions are nearly identical, with minimal difference
        </div>
        <div style="background-color:#eaf9d6; padding:5px; margin-bottom:5px;">
            <strong>0.6 - 0.8</strong>: High Similarity – Descriptions are very similar, with some differences
        </div>
        <div style="background-color:#fff9e6; padding:5px; margin-bottom:5px;">
            <strong>0.4 - 0.6</strong>: Moderate Similarity – Descriptions have noticeable differences, but share common topics
        </div>
        <div style="background-color:#ffe6cc; padding:5px; margin-bottom:5px;">
            <strong>0.2 - 0.4</strong>: Low Similarity – Descriptions have overlapping content, but are generally quite different
        </div>
        <div style="background-color:#ffd6cc; padding:5px; margin-bottom:5px;">
            <strong>0.0 - 0.2</strong>: Very Low Similarity – Descriptions are largely different with little to no overlap
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if sending_course_desc and university != "Select...":
            try:
                courses_df = load_courses(university)
                subjects = courses_df['Subject'].unique().tolist()
                vectorizer, subject_vectors = fit_subject_vectorizer(subjects)
                relevant_subjects, _, _ = identify_relevant_subjects(sending_course_desc, subjects, vectorizer, subject_vectors)

                filtered_df = courses_df[courses_df['Subject'].isin(relevant_subjects)]
                filtered_courses = dict(zip(filtered_df['Course Title'], filtered_df['Description']))
                course_embeddings = get_course_embeddings(filtered_courses)

                top_10_courses = compare_courses_batch(sending_course_desc, filtered_courses, cached_embeddings=course_embeddings)

                # Display results
                st.subheader("Relevant Subjects")
                st.write(", ".join(relevant_subjects) if relevant_subjects else "No relevant subjects found.")

                st.subheader(f"Top 10 Most Similar {university} Courses")
                for title, score in top_10_courses:
                    st.markdown(f"""
                    <div style="background-color:{get_color(score)}; padding:10px; margin-bottom:5px;">
                        <strong>{title}</strong> (Similarity Score: {score:.2f})
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error loading courses: {e}")
        else:
            st.warning("Please enter a course description and select a university.")

        # --- Disclaimer ---
        st.markdown("<h3 style='color:#1e3d58;'>Disclaimer</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color:#f0f4f8; padding: 15px; border-radius: 10px; color: #1e3d58;">
            <p>This tool is not an indicator of whether the sending course is/will be credited as one of the courses from the receiving university. 
            It's simply a starting point for students to petition for credit or for universities to easily assess which courses could potentially be assigned credit.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
