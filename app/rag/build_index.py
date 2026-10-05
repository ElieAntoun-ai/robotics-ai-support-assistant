from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Extract the content of the pdfs
#--------------------------------------------------------------
documents = []

pdf_files = Path("data/documents").glob("*.pdf")

for pdf_file in pdf_files:
    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())

print(f"Number of pages loaded: {len(documents)}")
#-------------------------------------------------------------

#PERFORM THE CHUNKING ON THE DOCUMENTS EXTRACTED FROM THE PDFS
#-------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")
#-----------------------------------------------------------------------

#INITIALIZE THE EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#CREATE THE VECTOR STORE AND SAVE THE VECTOR INDEX LOCALLY
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

vector_store.save_local("faiss_index")

print("FAISS index created and saved successfully!")
#------------------------------------------------------------------------------------