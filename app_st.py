import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv(override=True)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "dts-project-data"
index = pc.Index(index_name)

# Optional: Auto-create index if not exists
if index_name not in [i['name'] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/gtr-t5-base",
        model_kwargs={"device": "cpu"}
    )
    return PineconeVectorStore(index=index, embedding=embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

st.title("🚀 Simple Streamlit Module Bot")

with st.sidebar:
    groq_api_key = st.text_input("GROQ API Key", type="password")
    "[Get GROQ API key](https://console.groq.com/keys)"


def generate_response(input_text):
    model = 'llama-3.3-70b-versatile'
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model    
    )
 
    # Define a function to format the retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    context = format_docs(retriever.invoke(input_text))

    # Define the prompt template for generating AI responses
    PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    The response should be specific and use statistics or numbers when possible.
    Please answer with the same language as the question.

    Assistant:"""

    PROMPT_TEMPLATE = PROMPT_TEMPLATE.replace("{context}", context)

    # Create a PromptTemplate instance with the defined template and input variables
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["question"]
    )

    # Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
    chain = (
        # {"question": RunnablePassthrough()}
        prompt
        | groq_chat
        | StrOutputParser()
    )

    st.info(chain.invoke({"question": input_text}))

with st.form("my_form"):
    text = st.text_area("Enter text:", "Who are the head master of Hogwrats?")
    submitted = st.form_submit_button("Submit")
    if not groq_api_key:
        st.info("Please add your GROQ API key to continue.")
    elif submitted:
        generate_response(text)
