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

# Embedding and Vector Store
embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
db=FAISS.from_documents(pages,embeddings)
retriever=db.as_retriever()

llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",google_api_key=GOOGLE_API_KEY)
memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# form state
if "messages" not in st.session_state:
    st.session_state.messages=[]

if"form_state " not in st.session_state:
    st.session_state.form_state={"step":0,"name":"","email":"","phone":"","date":""}

# Streamlit UI
st.title("Chatbot: Ask & Book")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input=st.chat_input("ask something")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

intent_keywords=["call","book","appointment","schedule"]
#intent detection of the user_input
intent_triggered=any(word in user_input.lower() for word in intent_keywords)


    response = qa_chain.run(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)