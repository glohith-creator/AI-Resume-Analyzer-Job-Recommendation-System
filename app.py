import os
import streamlit as st
import pandas as pd

from resume_engine import (
    load_skill_dictionary,
    load_job_roles,
    process_resume,
    generate_roadmap
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .score-card {
        padding: 18px;
        border-radius: 12px;
        background-color: #f4f6f8;
        text-align: center;
        margin-bottom: 10px;
    }

    .score-number {
        font-size: 32px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SKILLS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "skill_dictionary.csv"
)

ROLES_FILE = os.path.join(
    BASE_DIR,
    "data",
    "job_roles.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

try:

    skill_dictionary = load_skill_dictionary(
        SKILLS_FILE
    )

    job_roles = load_job_roles(
        ROLES_FILE
    )

except Exception as error:

    st.error(
        f"Could not load project data: {error}"
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📄 AI Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Analyze your resume, identify skills, discover suitable
    job roles and find skills you can develop.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write(
        """
        **AI Resume Analyzer and Job Recommendation System**

        This system uses:

        • Resume text extraction  
        • Skill extraction  
        • TF-IDF similarity  
        • Cosine similarity  
        • Skill-gap analysis  
        • Learning roadmap
        """
    )

    st.divider()

    st.subheader("Available Job Roles")

    for role in job_roles:
        st.write(
            "•",
            role["role"]
        )

    st.divider()

    st.caption(
        "Match scores are estimates based on the uploaded "
        "resume and predefined role profiles. They are not "
        "hiring decisions."
    )


# =========================================================
# UPLOAD SECTION
# =========================================================

st.subheader("1️⃣ Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"],
    help="Supported formats: PDF and DOCX"
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded: **{uploaded_file.name}**"
    )

    analyze_button = st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        with st.spinner(
            "Extracting and analyzing your resume..."
        ):

            try:

                analysis = process_resume(
                    uploaded_file,
                    skill_dictionary,
                    job_roles
                )

                st.session_state["analysis"] = analysis

                st.success(
                    "Resume analysis completed!"
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )


# =========================================================
# RESULTS
# =========================================================

if "analysis" in st.session_state:

    analysis = st.session_state["analysis"]

    detected_skills = analysis[
        "detected_skills"
    ]

    results = analysis[
        "results"
    ]

    top_roles = analysis[
        "top_roles"
    ]

    # -----------------------------------------------------
    # DETECTED SKILLS
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "2️⃣ Detected Skills"
    )

    if detected_skills:

        columns = st.columns(4)

        for index, skill in enumerate(
            detected_skills
        ):

            columns[
                index % 4
            ].success(skill)

    else:

        st.warning(
            "No recognized skills were detected "
            "from the uploaded resume."
        )


    # -----------------------------------------------------
    # ROLE MATCH SCORES
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "3️⃣ Job Role Match Scores"
    )

    chart_data = pd.DataFrame(
        {
            "Job Role": [
                result["role"]
                for result in results
            ],
            "Match Score": [
                result["score"]
                for result in results
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index("Job Role")
    )


    # -----------------------------------------------------
    # TOP 3 RECOMMENDATIONS
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "4️⃣ Top 3 Recommended Job Roles"
    )

    for index, result in enumerate(
        top_roles,
        start=1
    ):

        col1, col2 = st.columns(
            [1, 3]
        )

        with col1:

            st.markdown(
                f"""
                <div class="score-card">

                <div>#{index}</div>

                <div class="score-number">
                {result["score"]}%
                </div>

                <div>Match Score</div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.subheader(
                result["role"]
            )

            if result["matched_skills"]:

                st.write(
                    "**Matched Skills:** "
                    +
                    ", ".join(
                        result["matched_skills"]
                    )
                )

            else:

                st.write(
                    "**Matched Skills:** None detected"
                )

        st.divider()


    # -----------------------------------------------------
    # TARGET ROLE
    # -----------------------------------------------------

    st.subheader(
        "5️⃣ Detailed Skill Gap Analysis"
    )

    role_names = [
        result["role"]
        for result in results
    ]

    selected_role = st.selectbox(
        "Select a target job role",
        role_names
    )

    selected_result = next(
        result
        for result in results
        if result["role"] == selected_role
    )


    # -----------------------------------------------------
    # MATCH SCORE
    # -----------------------------------------------------

    st.metric(
        "Match Score",
        f'{selected_result["score"]}%'
    )


    # -----------------------------------------------------
    # MATCHED / MISSING SKILLS
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.markdown(
            "### ✅ Skills You Have"
        )

        if selected_result[
            "matched_skills"
        ]:

            for skill in selected_result[
                "matched_skills"
            ]:

                st.write(
                    f"✅ {skill}"
                )

        else:

            st.info(
                "No matching skills detected."
            )


    with right:

        st.markdown(
            "### ⚠️ Skills to Develop"
        )

        if selected_result[
            "missing_skills"
        ]:

            for skill in selected_result[
                "missing_skills"
            ]:

                st.write(
                    f"❌ {skill}"
                )

        else:

            st.success(
                "No major missing skills detected."
            )


    # -----------------------------------------------------
    # LEARNING ROADMAP
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "6️⃣ Suggested Learning Roadmap"
    )

    roadmap = generate_roadmap(
        selected_result
    )

    if roadmap:

        for index, step in enumerate(
            roadmap,
            start=1
        ):

            st.write(
                f"**Step {index}:** {step}"
            )

    else:

        st.info(
            "No roadmap available."
        )


    # -----------------------------------------------------
    # EXTRACTED RESUME TEXT
    # -----------------------------------------------------

    st.divider()

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        st.text(
            analysis["resume_text"]
        )


    # -----------------------------------------------------
    # METHODOLOGY
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "ℹ️ Methodology"
    )

    st.write(
        """
        The system combines TF-IDF cosine similarity and
        skill overlap to calculate an estimated match score.

        The system does not make hiring decisions and does
        not evaluate protected personal attributes.
        """
    )


# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.info(
        "Upload a PDF or DOCX resume above and click "
        "**Analyze Resume** to begin."
    )