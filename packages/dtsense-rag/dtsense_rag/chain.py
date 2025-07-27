import os

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv(dotenv_path="D:/DTSense/DHS_MODULE_ST_BOT-main/.env")

# Get Groq API key
#groq_api_key = os.environ['GROQ_API_KEY']
model = 'llama-3.3-70b-versatile'
# Initialize Groq Langchain chat object and conversation
groq_chat = ChatGroq(
        groq_api_key='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', 
        model_name=model
)
# response = groq_chat.invoke("Who are the authors of DTSense Streamlit Book?")
# print(response)


# basic_chain = prompt | groq_chat | StrOutputParser()
# response = basic_chain.invoke({"context": "", "question": "Who are the authors of DTSense Streamlit Book?"})
# print(response)

# embedding = HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
#     )

# URI = "packages/dtsense-rag/dtsense_rag/data/milvus_lite.db"

# vectordb = Milvus(
#     embedding_function=HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
#     ),
#     collection_name="dtsense_streamlit",
#     connection_args={"uri": URI},
# )
# retriever = vectordb.as_retriever()

current_dir = os.path.dirname(__file__)
sample_path = os.path.join(current_dir, 'data', 'sample.txt')

context = ""

#with open("packages/dtsense-rag/dtsense_rag/data/sample.txt") as f:
with open(sample_path, 'r', encoding='utf-8') as f:
    context = f.read()

# loader = TextLoader("./index.md")
# loader.load()


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

# response = rag_chain.invoke("Siapa dari the authors dari module DTSense Streamlit?")
# print(response)
