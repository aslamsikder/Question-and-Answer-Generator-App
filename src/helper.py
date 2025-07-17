
from langchain.document_loaders import PyPDFLoader
from transformers import AutoTokenizer
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI # If you use open AI API
#from langchain_openrouter import ChatOpenRouter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from src.prompt import *


#API Authentication
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# Data Preprocessing Function has been started here
def file_processing(file_path):

    #Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ""

    for page in data:
        question_gen += page.page_content

    # Load Mistral tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    use_fast=False
    )

    #First Layer Chunking
    splitter_ques_gen = TokenTextSplitter.from_huggingface_tokenizer(
    tokenizer = tokenizer,
    chunk_size = 10000,
    chunk_overlap = 200
    )

    chunk_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunk_ques_gen]

    #Second Layer Chunking
    splitter_ans_gen = TokenTextSplitter.from_huggingface_tokenizer(
    tokenizer = tokenizer,
    chunk_size = 1000,
    chunk_overlap = 100
    )

    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


# LLM Pipeline Function has been started here
def llm_pipeline (file_path):
    
    document_ques_gen, document_answer_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatOpenAI(
    model_name = 'mistralai/Mistral-7B-Instruct',
    temperature = 0.3,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1"
    )

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=['text'])

    REFINED_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refined_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline,
                                      chain_type="refine",
                                      verbose=True,
                                      question_prompt = PROMPT_QUESTIONS,
                                      refine_prompt=REFINED_PROMPT_QUESTIONS)
    
    ques = ques_gen_chain.run(document_ques_gen)

    # Embedding has been done with hugging face model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Used FAISS as a Vector Database
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = ChatOpenAI(
    model_name = 'mistralai/Mistral-7B-Instruct',
    temperature = 0.1,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1"
    )

    ques_list = ques.split("\n")

    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen,
                                                      chain_type="stuff",
                                                      retriever=vector_store.as_retriever())
    
    return answer_generation_chain, filtered_ques_list


