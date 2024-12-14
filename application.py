import json
import sys
# Ensure compatibility with pysqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
from src.get_answers import get_answers
from src.process_doc import load_data

# Streamlit app title
st.title("PDF Question Answering App")

# File uploader for PDF files
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Process the uploaded file
if uploaded_file is not None:
    # Ensure the 'docs' directory exists
    if not os.path.exists('docs'):
        os.makedirs('docs')

    # Save the uploaded file to the 'docs' directory
    file_path = os.path.join('docs', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load data from the PDF file
    with st.spinner("Extracting text from PDF..."):
        load_data(uploaded_file.name)

# Text area for entering questions
questions_input = st.text_area(
    "Enter your questions (one per line)",
    value="""What is the name of the company?
Who is the CEO of the company?
What is their vacation policy?
What is the termination policy?"""
)

# Button to trigger answer retrieval
if st.button("Get Answers"):
    if questions_input:
        # Split the input into individual questions
        questions = questions_input.strip().split("\n")

        # Display a subheader for answers
        st.subheader("Answers")

        # Retrieve answers using the AI model
        with st.spinner("Getting answers from OpenAI..."):
            response = get_answers(questions)
            # Convert JSON response to a dictionary
            answers_dict = json.loads(response)

            # Display each question and its corresponding answer
            for question, answer in answers_dict.items():
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {answer}")
                st.write("---")  # Separator between Q&A pairs
    else:
        st.error("Please upload a PDF file and enter at least one question.")
