import json
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma
from src.process_doc import PERSIST_DIRECTORY

from concurrent.futures import ThreadPoolExecutor


def process_question(question):
    queries = get_query(question)
    relevant_docs = search_docs(queries)
    answer = get_answer(question, relevant_docs)
    return question, answer


def get_answers(questions: List[str]):
    results = {}
    with ThreadPoolExecutor(max_workers=len(questions)) as executor:
        future_to_question = {executor.submit(
            process_question, question): question for question in questions}
        for future in future_to_question:
            question, answer = future.result()
            results[question] = answer
    return json.dumps(results, indent=4)


def get_query(question: str) -> str:
    # First, based on the question, create a good query(s).
    prompt = f"""Given the input question, your job is to create a useful query(s) to search the document for the answer.
Your output should be a JSON object with the following structure. 
{{
    "query": "the query you created"
}}
If you cannot create a query, return the input question. The query should be as detailed as possible, to make the search more accurate.

Input question: {question}"""
    response = ChatOpenAI(
        model='gpt-4o-mini').invoke([SystemMessage(prompt), HumanMessage(question)], response_format={"type": "json_object"})

    # Parse the response to get the queries.
    try:
        response = json.loads(response.content)
        query = response['query']
        return [query]
    except json.JSONDecodeError:
        return [question]


def search_docs(queries: List[str]) -> List[str]:
    relevant_docs = []
    for query in queries:
        docs = Chroma(collection_name='handbook.pdf', persist_directory=PERSIST_DIRECTORY,
                      embedding_function=OpenAIEmbeddings()).similarity_search(query=query, k=4,)
        relevant_docs.extend(docs)
    return relevant_docs


def get_answer(question: str, relevant_docs: List[str]) -> str:
    # Now, based on the relevant documents, find the answer to the question.
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
    response = ChatOpenAI(
        model='gpt-4o-mini').invoke([SystemMessage(prompt)])

    return response.content
