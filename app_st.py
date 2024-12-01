import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("🚀 DTSense Streamlit Module Bot")

with st.sidebar:
    groq_api_key = st.text_input("GROQ API Key", type="password")
    "[Get GROQ API key](https://console.groq.com/keys)"


def generate_response(input_text):
    model = 'llama-3.2-11b-vision-preview'
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model    
    )

    with open("packages/dtsense-rag/dtsense_rag/data/sample.txt", encoding="utf-8") as f:
        context = f.read()

    # Define a function to format the retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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
    text = st.text_area("Enter text:", "Who are the authors of DTSense Streamlit Module?")
    submitted = st.form_submit_button("Submit")
    if not groq_api_key:
        st.info("Please add your GROQ API key to continue.")
    elif submitted:
        generate_response(text)