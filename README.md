# AI-Powered Tailored Resume Generator

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)

The **AI-Powered Tailored Resume Generator** is a Streamlit-based web application that helps users optimize their resumes for specific job descriptions. It uses natural language processing (NLP) and machine learning to compare resumes with job descriptions, provide keyword-based suggestions, and generate tailored resumes.

## Features

- **User Authentication**: Sign up and log in with email and password. Passwords are securely hashed and stored in a SQLite database.
- **Resume Upload**: Upload resumes in PDF or DOCX format.
- **Keyword Extraction**: Extract keywords from resumes and job descriptions.
- **ATS Score Calculation**: Calculate the Applicant Tracking System (ATS) compatibility score for your resume.
- **Resume Comparison**: Compare your resume with a job description to identify matched and mismatched keywords.
- **Tailored Resume Generation**: Automatically update your resume with missing keywords and download the updated version.
- **Visualizations**: View pie charts for keyword matching and overall similarity.

## Demo

![Demo](https://raw.githubusercontent.com/your_username/your_repo/main/static/demo.gif)  
*Replace with a link to your demo GIF or video.*

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/your_repo.git
   cd your_repo

2.Install Dependencies:

pip install -r requirements.txt


3.Run the Application:

streamlit run app.py


4.Access the Application:
Open your browser and go to http://localhost:8501

5.Usage
    1.Sign Up: Create a new account by providing your name, email, and password.
    
    2.Log In: Log in with your registered email and password.
    
    3.Upload Resume: Upload your resume in PDF or DOCX format.
    
    4.Enter Job Description: Provide the job description you want to compare your resume with.
    
    5.View Results:
    
          *See matched and mismatched keywords.
          
          *Check your ATS score.
          
          *Get suggestions to improve your resume.
    
    6.Download Updated Resume: Download the updated resume with added keywords.


File Structure

your_repo/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── users.db                # SQLite database for user authentication
├── static/                 # Static files (e.g., images, demo GIF)
│   └── demo.gif
└── README.md               # Project documentation


Technologies Used
Streamlit: For building the web application.

PyMuPDF (fitz): For extracting text and images from PDF files.

LangChain: For text splitting, embeddings, and conversational retrieval.

FAISS: For efficient similarity search.

SQLite: For storing user data.

Python-docx: For handling DOCX files.

Matplotlib: For generating visualizations.
