import os
import re
import dateparser
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
import streamlit as st

# Rag system
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
assert GOOGLE_API_KEY, "GOOGLE_API_KEY not found in .env file"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# a class for appointmet that the agent uses
class Appointment:
    def __init__(self):
        self.state = {"step": 0, "name": "", "email": "", "phone": "", "date": ""}

    def reset(self):
        self.state = {"step": 0, "name": "", "email": "", "phone": "", "date": ""}

    def run(self, user_input: str) -> str:
        step = self.state["step"]
        if step == 0:
            self.state["step"] = 1
            return "Sure! Please enter your full name."

        elif step == 1:
            self.state["name"] = user_input
            self.state["step"] = 2
            return "Enter your email"

        elif step == 2:
            if re.match(r"[a-zA-Z0-9_.+-]+@[a-zA-Z]+\.[a-zA-Z]+$", user_input):
                self.state["email"] = user_input
                self.state["step"] = 3
                return "Thanks! What's your phone number?"
            else:
                return "Please enter a valid email."

        elif step == 3:
            if re.match(r"^\+?\d{7,15}$", user_input):
                self.state["phone"] = user_input
                self.state["step"] = 4
                return "And what date do you want to book? (e.g., year-month-day or next Monday)"
            else:
                return "Please enter a valid phone number."

        elif step == 4:
            parsed_date = dateparser.parse(user_input)
            if parsed_date:
                self.state["date"] = parsed_date.strftime("%Y-%m-%d")
                self.state["step"] = 5
                return (
                    f" Appointment booked!\n\n"
                    f"**Name:** {self.state['name']}\n"
                    f"**Email:** {self.state['email']}\n"
                    f"**Phone:** {self.state['phone']}\n"
                    f"**Date:** {self.state['date']}"
                )
            else:
                return "Sorry,couldn't understand that date. Try 'next Monday' or '2025-07-15'."

        else:
            return "Appointment already booked. Say 'restart' to start over."

appointment_form = Appointment()

@tool(description="Handles conversational booking of appointments (name, email, phone, date).")
def appointment_tool(user_input: str) -> str:
    if user_input.lower().strip() == "restart":
        appointment_form.reset()
        return "Appointment form restarted. Please enter your full name."
    return appointment_form.run(user_input)

# PDF LOADING 
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # one Document per page

    # Split into smaller chunks 
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever()

qa_chain = None

@tool
def pdf_qa_tool(question: str) -> str:
    """Answers user questions using the uploaded PDF document."""
    if qa_chain is None:
        return "Please upload a PDF first."
    return qa_chain.run(question)

agent = None

# === STREAMLIT UI ===
st.title("Chatbot: Ask Questions & Book Appointment")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    retriever = load_pdf("temp.pdf")
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    tools = [appointment_tool, pdf_qa_tool]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    st.success("PDF uploaded and ready for questions!")

# streamlit session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Ask something or book an appointment...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    if agent is None:
        response = "Please upload a PDF first."
    else:
        response = agent.run(user_input)

    st.session_state["messages"].append({"role": "assistant", "content": response})

# === DISPLAY CHAT HISTORY ===
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
