from langchain_community.document_loaders import ArxivLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def arxiv_loader(paper_id: str) -> list[Document]:
    docs = ArxivLoader(query=paper_id, load_max_docs=2).load()
    return docs

def chunk_data(docs, chunk_size=800, chunk_overlap=50) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    return split_docs
