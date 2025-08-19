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
    st.write("TF-IDF Similarities:", {subject: round(sim, 3) for subject, sim in zip(subjects, similarities)})
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

        sending_course_desc = st.text_area("Enter the description for the sending university course")
        university = st.selectbox("Select the receiving university", ["Select...", "Pennsylvania State University", "Temple University", "West Chester University of PA"])

        st.markdown("""
        This tool compares a sending university course to courses at a receiving university using NLP to identify the top 10 most similar courses.
        """)

    with col2:
        if sending_course_desc and university != "Select...":
            try:
                # Load courses and subjects
                courses_df = load_courses(university)
                subjects = courses_df['Subject'].unique().tolist()
                vectorizer, subject_vectors = fit_subject_vectorizer(subjects)
                relevant_subjects, _, _ = identify_relevant_subjects(sending_course_desc, subjects, vectorizer, subject_vectors)

                # Filter courses by relevant subjects
                filtered_df = courses_df[courses_df['Subject'].isin(relevant_subjects)]
                filtered_courses = dict(zip(filtered_df['Course Title'], filtered_df['Description']))

                # Load cached embeddings for filtered courses
                course_embeddings = get_course_embeddings(filtered_courses)

                # Compute top 10 similar courses
                top_10_courses = compare_courses_batch(sending_course_desc, filtered_courses, cached_embeddings=course_embeddings)

                # Display relevant subjects and results
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

if __name__ == "__main__":
    main()
