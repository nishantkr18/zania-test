import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from src.get_answers import get_answers
from src.process_doc import load_data


# Streamlit app
st.title("PDF Question Answering App")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
# save the uploaded file to a local directory
if uploaded_file is not None:
    with open('docs/'+uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner("Extracting text from PDF..."):
        load_data(uploaded_file.name)

questions_input = st.text_area("Enter your questions (one per line)", value="""What is the name of the company?
Who is the CEO of the company?
What is their vacation policy?
What is the termination policy?""")

if st.button("Get Answers"):
    if questions_input:
        questions = questions_input.split("\n")
        st.subheader("Answers")
        with st.spinner("Getting answers from OpenAI..."):
            for pair in get_answers(questions):
                st.write("Question:", pair[0])
                st.write("Answer:", pair[1])

    else:
        st.error("Please upload a PDF file and enter at least one question.")
