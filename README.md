# Zania | AI Challenge

## Problem Statement

Create an AI agent that leverages the capabilities of a large language model. This agent should be able to extract answers based on the content of a large PDF document. Ideally, use OpenAI LLMs. If you use the Langchain or LLama Index framework to implement this agentic functionality, please don’t use pre-built chains for the task. Implement the logic yourself. Please write production-grade code as opposed to scripts, as we will be evaluating your code quality.

## Solution:

## Overview

This is an AI-powered application designed to extract answers from large PDF documents using OpenAI's language models. The application processes PDF files, creates vector embeddings, and retrieves answers to user-provided questions. The solution is built using the Langchain framework, with custom logic for document processing and question answering.

## Features

- **PDF Upload**: Users can upload PDF files for processing.
- **Question Answering**: Enter questions to retrieve answers based on the content of the uploaded PDF.
- **Vector Embeddings**: Utilizes Chroma for vector storage and retrieval.
- **Concurrent Processing**: Handles multiple questions simultaneously for efficient answer retrieval.

## Quick Start Guide

1. **Clone the Repository**:   ```bash
   git clone <repository-url>   ```

2. **Install Dependencies**:
   Navigate to the project directory and install the required packages:   ```bash
   pip install -r requirements.txt   ```

3. **Run the Application**:
   Start the Streamlit application:   ```bash
   streamlit run application.py   ```

4. **Upload a PDF**:
   Use the file uploader in the app to upload a PDF document.

5. **Ask Questions**:
   Enter your questions in the provided text area and click "Get Answers" to retrieve responses.

## Command-Line Usage

You can also use the `run.py` script to process a PDF and get answers to questions directly from the command line:

1. **Run the Script**:
   ```bash
   python run.py <pdf_filename> <question1> <question2> ... <questionN>
   ```

   - Replace `<pdf_filename>` with the path to your PDF file.
   - Replace `<question1>`, `<question2>`, ..., `<questionN>` with the questions you want to ask.

2. **Example**:
   ```bash
   python run.py docs/example.pdf "What is the main topic?" "Who is the author?"
   ```

   Ensure that `example.pdf` is located inside the `docs` folder. This will output the answers to the provided questions based on the content of `example.pdf`.


## Code Structure

- **[application.py](application.py)**: The main entry point for the Streamlit application. Handles file uploads and user interactions.
  - References: [View Code](application.py#L1-L63)

- **[src/process_doc.py](src/process_doc.py)**: Contains functions for loading PDF data, creating document chunks, and storing vector embeddings.
  - References: [View Code](src/process_doc.py#L1-L105)

- **[src/get_answers.py](src/get_answers.py)**: Implements the logic for retrieving answers using the vector database and AI model.
  - References: [View Code](src/get_answers.py#L1-L79)

- **[requirements.txt](requirements.txt)**: Lists all the dependencies required to run the application.
  - References: [View Code](requirements.txt#L1-L166)

- **[.gitignore](.gitignore)**: Specifies files and directories to be ignored by Git.
  - References: [View Code](.gitignore#L1-L174)

## Future Enhancements

- **Accuracy Improvements**: Explore advanced techniques for improving the accuracy of the AI model's responses.
- **Scalability**: Implement distributed processing for handling larger datasets and more concurrent users.
- **User Interface**: Enhance the UI for a better user experience.

## License

This project is licensed under the MIT License.

## Contact

For any questions or feedback, please contact [Nishant] at [nniishantkumar@gmail.com].


