from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

all_documents = list(
    vector_store.docstore._dict.values()
)
tokenized_documents = [
    document.page_content.lower().split()
    for document in all_documents
]

bm25 = BM25Okapi(tokenized_documents)

def retrieve_documents(question, k=5):
    results = vector_store.similarity_search(
        question,
        k=10
    )

    tokenized_question = question.lower().split()

    bm25_results = bm25.get_top_n(
    tokenized_question,
    all_documents,
    n=10
)
    combined_results = results + bm25_results
    unique_results = []

    for result in combined_results:
     if result not in unique_results:
        unique_results.append(result)
    
    pairs = [
    [question, result.page_content]
    for result in unique_results
]
    scores = reranker.predict(pairs)
    scored_results = list(zip(unique_results, scores))

    scored_results.sort(
    key=lambda x: x[1],
    reverse=True
)   
    reranked_results = [
    result
    for result, score in scored_results[:k]
]

    return reranked_results