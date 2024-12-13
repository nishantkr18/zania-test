import logging
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# Set logging level to INFO for detailed output
logging.basicConfig(level=logging.INFO)

# Load environment variables, including OpenAI API key
load_dotenv()

# Constants for Chroma vector database
PERSIST_DIRECTORY = "./.chroma_db"
COLLECTION_NAME = "collection"

def check_collection_exists(file_name: str) -> bool:
    """
    Check if a collection with the given file name already exists in the vectorstore.

    Args:
        file_name (str): The name of the file to check in the vectorstore.

    Returns:
        bool: True if the collection exists, False otherwise.
    """
    chroma_instance = Chroma(
        collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIRECTORY)
    metadata_list = chroma_instance.get(include=['metadatas'])['metadatas']
    return any(file_name in data['source'] for data in metadata_list)


def create_chunks(docs: List[Document], splitter: CharacterTextSplitter | None = None) -> List[Document]:
    """
    Split documents into smaller chunks for processing.

    Args:
        docs (List[Document]): List of documents to be split.
        splitter (CharacterTextSplitter, optional): Text splitter configuration. Defaults to a newline separator with chunk size 800 and overlap 200.

    Returns:
        List[Document]: List of document chunks.
    """
    if splitter is None:
        splitter = CharacterTextSplitter(
            separator="\n", chunk_size=800, chunk_overlap=200)

    source_chunks = splitter.split_documents(docs)
    logging.info(f'Chunks created: {len(source_chunks)}')

    max_length_document = max(
        source_chunks, key=lambda doc: len(doc.page_content))
    logging.info(
        f'Max length document has {len(max_length_document.page_content)} characters.')

    return source_chunks

def process_chunks(source_chunks: List[Document]) -> Chroma:
    """
    Create vector embeddings from document chunks and store them in the vectorstore.

    Args:
        source_chunks (List[Document]): List of document chunks to process.

    Returns:
        Chroma: The vectorstore instance containing the embeddings.
    """
    logging.info('Creating and storing embeddings in vectorstore...')
    vectorstore = Chroma.from_documents(
        documents=source_chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )
    return vectorstore

def load_data(file_name: str = 'handbook.pdf', overwrite: bool = False):
    """
    Load data from a PDF file, process it into chunks, and store in the vectorstore.

    Args:
        file_name (str, optional): The name of the PDF file to load. Defaults to 'handbook.pdf'.
        overwrite (bool, optional): Whether to overwrite existing data in the vectorstore. Defaults to False.
    """
    if not overwrite and check_collection_exists(file_name):
        logging.info(
            'Collection already exists in vectorstore. No need to load again.')
        return

    # Load documents from the specified PDF file
    loader = PyPDFLoader(f'docs/{file_name}')
    try:
        docs = loader.load()
        logging.info(f"{len(docs)} documents loaded.")
    except Exception as e:
        logging.error(f"Error loading documents: {e}")
        raise

    # Create chunks and process them into the vectorstore
    chunks = create_chunks(docs)
    process_chunks(chunks)
    logging.info('Created vectorstore!')
