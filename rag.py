import os
import re
import dateparser
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY, "GOOGLE_API_KEY not found in .env file"

# Load document
loader = PyPDFLoader("your_docs.pdf")
pages = loader.load()

# Embedding + retriever
embedding = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY)
db = FAISS.from_documents(pages, embedding)
retriever = db.as_retriever()

# LLM + Memory
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# Initialize form state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "form_state" not in st.session_state:
    st.session_state.form_state = {"step": 0, "name": "", "email": "", "phone": "", "date": ""}

# Intent keywords
intent_keywords = ["call", "appointment", "schedule", "book", "talk"]

# Streamlit UI
st.title("Chatbot: Ask & Book")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Ask something...")
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Check for intent
    intent_triggered = any(word in user_input.lower() for word in intent_keywords)

    # Begin form if triggered
    if intent_triggered or st.session_state.form_state["step"] > 0:
        step = st.session_state.form_state["step"]

        if step == 0:
            st.session_state.form_state["step"] += 1
            bot_msg = "Sure! What's your full name?"
        elif step == 1:
            st.session_state.form_state["name"] = user_input
            st.session_state.form_state["step"] += 1
            bot_msg = "Great. Can you share your email?"
        elif step == 2:
            # Email validation
            if re.match(r"[^@]+@[^@]+\.[^@]+", user_input):
                st.session_state.form_state["email"] = user_input
                st.session_state.form_state["step"] += 1
                bot_msg = "Thanks! What's your phone number?"
            else:
                bot_msg = "That doesn't look like a valid email. Please re-enter:"
        elif step == 3:
            # Phone validation
            if re.match(r"^\+?\d{7,15}$", user_input):
                st.session_state.form_state["phone"] = user_input
                st.session_state.form_state["step"] += 1
                bot_msg = "And what date do you want to book? (e.g., next Monday)"
            else:
                bot_msg = "Please enter a valid phone number:"
        elif step == 4:
            # Parse date
            parsed_date = dateparser.parse(user_input)
            if parsed_date:
                date_str = parsed_date.strftime("%Y-%m-%d")
                st.session_state.form_state["date"] = date_str
                st.session_state.form_state["step"] += 1
                # Booking confirmation
                data = st.session_state.form_state
                bot_msg = (
                    f"Appointment booked!\n\n"
                    f"**Name:** {data['name']}\n"
                    f"**Email:** {data['email']}\n"
                    f"**Phone:** {data['phone']}\n"
                    f"**Date:** {data['date']}"
                )
            else:
                bot_msg = "Sorry, I couldn't understand that date. Try something like 'tomorrow' or '2025-07-15'."
        else:
            bot_msg = "Appointment already booked. Type 'restart' to book another."

        # Save bot reply
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

        # Reset if user types restart
        if user_input.lower().strip() == "restart":
            st.session_state.form_state = {"step": 0, "name": "", "email": "", "phone": "", "date": ""}
    else:
        # Regular RAG QA mode
        response = qa_chain.run(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)  