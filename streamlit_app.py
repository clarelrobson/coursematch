# Load necessary packages
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Page configs
st.set_page_config(
    page_title="CourseMatch",  # New site name
    page_icon="https://github.com/clarelrobson/coursematch/blob/main/C.png",  # Logo png
    layout="wide",           # Optional: Layout customization
)

# Website aesthetic customizations
st.markdown("""
<style>
/* Title color */
h1 {
    color: #1c3144; /* navy blue for the title */
}

/* Subheader color */
h2, h3 {
    color: #1c3144; /* navy blue for headers */
}

/* Body text color */
body, div, p, li {
    color: #3a506b; /* Grayish dark navy blue for body text */
}

/* Input box text color */
    .stTextArea textarea, .stTextInput input {
        color: #3a506b !important; /* Same color as the body text */
        background-color: #f0f4f8 !important; /* Optional: Light background for the input box */
    }
 </style>
    
""", unsafe_allow_html=True)

# Initialize the NLP model (paraphrase-MiniLM-L3-v2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use gpu
model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=device)

# Compare the course description with courses from the selected university using the model
def compare_courses_batch(sending_course_desc, receiving_course_descs):
    # Encode the sending course description
    sending_course_vec = model.encode(sending_course_desc, convert_to_tensor=True, device=device)

    # Encode all receiving course descriptions in batch
    receiving_descriptions = list(receiving_course_descs.values())
    receiving_course_vecs = model.encode(receiving_descriptions, convert_to_tensor=True, device=device, batch_size=32)

    # Compute cosine similarities for all pairs at once
    similarity_scores = util.pytorch_cos_sim(sending_course_vec, receiving_course_vecs)

    # Create a dictionary of similarity scores
    results = {title: similarity_scores[0][i].item() for i, title in enumerate(receiving_course_descs.keys())}

    # Sort by similarity score (highest first) and return the top 10
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    return sorted_results[:10]

# Get color based on similarity score
def get_color(score):
    if score >= 0.8:
        return "#d0f0e9"  # blue/green
    elif score >= 0.6:
        return "#eaf9d6"  # yellow/green
    elif score >= 0.4:
        return "#fff9e6"  # yellow
    elif score >= 0.2:
        return "#ffe6cc"  # orange
    else:
        return "#ffd6cc"  # red

# Streamlit Interface
def main():
    # Create two columns
    col1, col2 = st.columns([1.5, 1]) # set column proportions

    with col1:
        st.markdown("""
        <h1 style = "display: flex; align-items: center;">
            Course Match
            <img src="https://raw.githubusercontent.com/clarelrobson/credit-comparison-site/main/C.png" width = 50 style = "margin-lef: 10px;" />
        </h1>
        """, unsafe_allow_html = True)
        
        st.markdown("""
        This tool helps you determine how a course at one university (the **sending university**) compares to courses offered at another university (the **receiving university**). 

        - **Sending University test**: The institution where the course you want to evaluate is offered. Enter the description of this course in the input box.
        - **Receiving University**: The institution where you want to see comparable courses. Select this university from the dropdown menu.

        By analyzing course descriptions using advanced [Natural Language Processing (NLP) techniques](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2), this tool identifies the top 10 most similar courses from the receiving university. Each result is scored to reflect how closely the course descriptions match.
        """)

        # User input for the sending course description
        sending_course_desc = st.text_area("Enter the description for the sending university course")

        # Dropdown to select the university
        university = st.selectbox("Select the receiving university", ["Select...", "Pennsylvania State University", "Temple University", "West Chester University of PA"])

        # Similarity Rating Explanation
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
        # Right Column: Results and Disclaimer
        if sending_course_desc and university != "Select...":
            # URLs for the university course CSV files
            psu_courses_file_url = "https://raw.githubusercontent.com/clarelrobson/credit-comparison-site/main/psu_courses_with_credits.csv"
            temple_courses_file_url = "https://raw.githubusercontent.com/clarelrobson/credit-comparison-site/main/temple_courses_with_credits.csv"
            wcu_courses_file_url = "https://raw.githubusercontent.com/clarelrobson/credit-comparison-site/main/wcu_courses.csv"

            # Load the selected university's course descriptions CSV
            try:
                if university == "Pennsylvania State University":
                    courses_file_url = psu_courses_file_url
                elif university == "Temple University":
                    courses_file_url = temple_courses_file_url
                elif university == "West Chester University of PA":
                    courses_file_url = wcu_courses_file_url

                courses_df = pd.read_csv(courses_file_url)

                # Check if the necessary columns are present
                required_columns = ['Course Title', 'Description']
                if not all(col in courses_df.columns for col in required_columns):
                    st.error(f"{university} courses CSV must contain the columns: {', '.join(required_columns)}.")
                    return

                # Prepare dictionaries for course titles and descriptions
                courses = dict(zip(courses_df['Course Title'], courses_df['Description']))

                # Compare the sending course description with the selected university's courses
                top_10_courses = compare_courses_batch(sending_course_desc, courses)

                # Display the results with the header
                st.subheader(f"Top 10 Most Similar {university} Courses")

                for course_title, score in top_10_courses:
                    st.markdown(f"""
                    <div style="background-color:{get_color(score)}; padding:10px; margin-bottom:5px;">
                        <strong>{course_title}</strong> (Similarity Score: {score:.2f})
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error loading courses: {e}")
        else:
            st.warning("Please enter a course description and select a university.")

        # Disclaimer heading
        st.markdown("<h3 style='color:#1e3d58;'>Disclaimer</h3>", unsafe_allow_html=True)
    
        # Disclaimer body text
        st.markdown("""
        <div style="background-color:#f0f4f8; padding: 15px; border-radius: 10px; color: #1e3d58;">
            <p>This tool is not an indicator of whether the sending course is/will be credited as one of the courses from the receiving university. 
            It's simply a starting point for students to petition for credit or for universities to easily assess which courses could potentially be assigned credit.</p>
        </div>
        """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
