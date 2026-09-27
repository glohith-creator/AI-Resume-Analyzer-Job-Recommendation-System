# AI Resume Analyzer and Job Recommendation System

## 1. Project Overview

The AI Resume Analyzer and Job Recommendation System is an NLP-based application that analyzes a user's resume and provides job-role recommendations based on the skills and information extracted from the resume.

The system extracts resume text from PDF or DOCX files, identifies technical skills, compares the resume with predefined job-role profiles, calculates estimated match scores, identifies missing skills, and generates a learning roadmap.

The application is built using Python and Streamlit.

---

## 2. Objectives

The main objectives of this project are:

- Extract text from PDF and DOCX resumes.
- Clean and preprocess extracted resume text.
- Identify technical skills from the resume.
- Compare resume content with predefined job roles.
- Calculate estimated job-role match scores.
- Recommend the top 3 suitable job roles.
- Identify matched and missing skills.
- Generate a learning roadmap for missing skills.
- Provide an interactive Streamlit dashboard.

---

## 3. Technologies Used

### Programming Language
- Python

### Libraries
- Streamlit
- Pandas
- NumPy
- PyPDF
- python-docx
- Scikit-learn

### Machine Learning / NLP Techniques
- TF-IDF Vectorization
- Cosine Similarity
- Keyword-based Skill Extraction
- Skill Overlap Analysis

---

## 4. System Workflow

```text
Resume Upload
      ↓
PDF / DOCX Text Extraction
      ↓
Text Cleaning
      ↓
Skill Extraction
      ↓
Load Job Role Profiles
      ↓
TF-IDF + Cosine Similarity
      ↓
Skill Matching
      ↓
Match Score Calculation
      ↓
Top 3 Job Recommendations
      ↓
Skill Gap Analysis
      ↓
Learning Roadmap