import sys
from src.process_doc import load_data
from src.get_answers import get_answers


def main(pdf_filename, questions):
    """
    Main function to process a PDF file and retrieve answers to questions.

    Args:
        pdf_filename (str): The path to the PDF file to be processed.
        questions (list): A list of questions to retrieve answers for.
    """
    try:
        # Process the PDF to create vector embeddings
        load_data(pdf_filename)
    except Exception as e:
        print(f"Error processing PDF file '{pdf_filename}': {e}")
        sys.exit(1)

    try:
        # Retrieve answers for each question
        answers = get_answers(questions)
        print(answers)
    except Exception as e:
        print(f"Error retrieving answers: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure the script is run with the correct number of arguments
    if len(sys.argv) < 3:
        print("Usage: python run.py <pdf_filename> <question1> <question2> ... <questionN>")
        sys.exit(1)

    # Extract the PDF filename and questions from command-line arguments
    pdf_filename = sys.argv[1]
    questions = sys.argv[2:]

    # Execute the main function
    main(pdf_filename, questions)
