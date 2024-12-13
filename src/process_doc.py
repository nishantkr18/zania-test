from tabnanny import check
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.document import Document
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
import logging

# Set logging level as info
logging.basicConfig(level=logging.INFO)

# Add OpenAI API key and environment to environment variables
load_dotenv()

PERSIST_DIRECTORY = "./.chroma_db"


def check_collection_exists(file_name: str) -> bool:
    for data in Chroma(collection_name=file_name, persist_directory=PERSIST_DIRECTORY).get(include=['metadatas'])['metadatas']:
        if file_name in data['source']:
            return True
    return False


def create_chunks(
        docs: List[Document],
        splitter: CharacterTextSplitter = CharacterTextSplitter(separator="\n",
                                                                chunk_size=800, chunk_overlap=200)
) -> List[Document]:
    # Split into sentences
    source_chunks = splitter.split_documents(docs)
    logging.info(f'chunks created: {len(source_chunks)}')
    max_length_document = max(
        source_chunks, key=lambda doc: len(doc.page_content))
    logging.info(
        f'Max length document has {len(max_length_document.page_content)} characters.')
    return source_chunks


def process_chunks(source_chunks: List[Document], file_name) -> Chroma:
    # Create vector embeddings and store in vectorstore.
    logging.info('Creating and storing embeddings in vectorstore...')
    vectorstore = Chroma.from_documents(
        documents=source_chunks, embedding=OpenAIEmbeddings(), persist_directory=PERSIST_DIRECTORY, collection_name=file_name)
    return vectorstore


def load_data(file_name: str = 'handbook.pdf', overwrite: bool = False):
    # Check if collection exists in vectorstore, unless overwrite is True
    if not overwrite and check_collection_exists(file_name):
        logging.info(
            'Collection already exists in vectorstore. No need to load again.')
        return

    # Load documents from PDF
    loader = PyPDFLoader('docs/'+file_name)
    try:
        docs = loader.load()
        logging.info(f"{len(docs)} documents loaded.")
    except Exception as e:
        logging.error(f"Error loading documents: {e}")
        raise
    chunks = create_chunks(docs)
    process_chunks(chunks, file_name)
    logging.info('Created vectorstore!')


if __name__ == '__main__':
    load_data()
