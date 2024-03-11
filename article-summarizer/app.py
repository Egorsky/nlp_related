import arxiv
import openai
import langchain
import pinecone
from langchain_community.document_loaders import ArxivLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from utils import *
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
environment = os.getenv('PINECONE_ENV')

llm_summary = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo-0125")    
llm = OpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.6, api_key=openai_api_key)

if 'summary' not in st.session_state:
    st.session_state.summary = None

if 'documents' not in st.session_state:
    st.session_state.documents = None


st.title('Arxiv Paper Summarizer and Interactive Q&A')

paper_id_input = st.text_input('Enter Arxiv Paper ID', '')

if st.button('Summarize Paper') and paper_id_input:
    with st.spinner('Fetching and summarizing the paper...'):
        try:
            doc = arxiv_loader(paper_id=paper_id_input)
            st.session_state.documents = chunk_data(docs=doc)
#            st.write(st.session_state.documents)
            chain = load_summarize_chain(
                llm=llm_summary,
                    chain_type='map_reduce',
                    verbose=False
                    )
            summary = chain.run(st.session_state.documents)
            st.subheader('Summary')
            st.write(summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            

def initialize_index(index_name='arxiv-summarizer'):
#    documents = chunk_data(docs=doc)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    index_name = index_name
    # Make sure environment is correctly spelled (there was a typo in your provided code)
    pinecone.Pinecone(
        api_key=pinecone_api_key,
        environment=environment
    )
    if st.session_state.documents:
        index = Pinecone.from_documents(st.session_state.documents, embeddings, index_name=index_name)
    else:
        index = None
    return index

index = initialize_index()

def retrieve_query(query, k=2):
    matching_results = index.similarity_search(query, k=k)
    return matching_results

def retrieve_answers(query):
    chain = load_qa_chain(llm, chain_type='stuff')
    doc_search = retrieve_query(query)
    print(doc_search)
    response = chain.run(input_documents=doc_search, question=query)
    return response

if paper_id_input:
    user_query = st.text_input("Ask a question about the paper:", '')
    
    if user_query:
        if st.button('Get Answer'):
            with st.spinner('Retrieving your answer...'):
                try:
                    answer = retrieve_answers(user_query)
                    st.subheader('Answer')
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred while retrieving the answer: {e}")


