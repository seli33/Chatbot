import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY",None)
assert GOOGLE_API_KEY, "GOOGLE_API_KEY not found in .env file"
loader=PyPDFLoader("linear_regression.pdf")
pages=loader.load()
print(pages[13].page_content)
# Embedding and Vector Store
embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
db=FAISS.from_documents(pages,embeddings)
retriever=db.as_retriever()
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",google_api_key=GOOGLE_API_KEY)
memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
# Streamlit Interface
st.title("Document QA Chatbot")
user_input=st.text_input("ask something")

if user_input:
    response=qa_chain.run(user_input)
    st.write("bot:",response)