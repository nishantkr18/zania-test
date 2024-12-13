import json
from typing import List
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from src.process_doc import PERSIST_DIRECTORY, COLLECTION_NAME
from concurrent.futures import ThreadPoolExecutor


def get_answers(questions: List[str]) -> str:
    """
    Retrieves answers for a list of questions using a vector database and an AI model.

    Args:
        questions (List[str]): A list of questions to be answered.
        file_name (str): The name of the file associated with the questions.

    Returns:
        str: A JSON string containing the questions and their corresponding answers.
    """
    # Initialize the Chroma vector database with the specified collection and embeddings
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=OpenAIEmbeddings()
    )

    results = {}
    # Use a ThreadPoolExecutor to handle multiple questions concurrently
    with ThreadPoolExecutor(max_workers=len(questions)) as executor:
        # Map each question to a future that will hold its answer
        future_to_question = {
            executor.submit(get_answer, db, question): question for question in questions
        }
        # Collect results as they are completed
        for future in future_to_question:
            question, answer = future.result()
            results[question] = answer

    # Return the results as a formatted JSON string
    return json.dumps(results, indent=4)


def get_answer(db: Chroma, question: str) -> tuple[str, str]:
    """
    Finds the answer to a question using relevant documents from a vector database.

    Args:
        db (Chroma): The vector database instance.
        question (str): The question to be answered.

    Returns:
        tuple: A tuple containing the question and its corresponding answer.
    """
    # Perform a similarity search to find relevant documents for the question
    relevant_docs = db.similarity_search(query=question, k=5)

    # Construct a prompt for the AI model to generate an answer
    prompt = f"""Given the input question and relevant documents, your job is to find the answer to the question.

Your output should meet the following criteria:
1. If the question matches a sentence or phrase word-for-word in the relevant documents, extract that exact sentence or phrase as the answer.
2. If the question does not match word-for-word but the answer can still be inferred confidently from the relevant documents, provide the most accurate answer supported by the documents.
3. If you cannot confidently determine the answer from the relevant documents, respond with "Data Not Available."

Input:
Question: {question}
Relevant Documents: {relevant_docs}

Note:
- You are allowed to correct any grammatical errors in the answer. But make sure that the meaning of the answer is preserved.
- Do not hallucinate or generate answers that are not supported by the documents.
"""

    # Invoke the AI model with the constructed prompt
    response = ChatOpenAI(model='gpt-4o-mini').invoke([SystemMessage(prompt)])

    # Return the question and the generated answer
    return question, response.content
