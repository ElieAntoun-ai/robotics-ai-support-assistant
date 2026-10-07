from langchain_core.tools import tool
from app.rag.retriever import retrieve_documents
from pathlib import Path


@tool
def search_documentation(question: str):
    """
    Search the Husky A300 technical documentation for specifications,
    operating instructions, maintenance information, troubleshooting,
    and other technical information.
    """

    results = retrieve_documents(question)

    context = "\n\n".join(
        f"""
Content:
{result.page_content}

Source: {Path(result.metadata['source']).name}
Page: {result.metadata['page_label']}
"""
        for result in results
    )

    return context