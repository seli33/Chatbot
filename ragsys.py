import os
import re
import dateparser
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

uploaded_file=st.file_uploader("Upload a pdf document",type="pdf")
if uploaded_file is not None:
    with open("temp_uploaded.pdf","wb") as f:
        f.write(uploaded_file.read())

    loader=PyPDFLoader("temp_uploaded.pdf")
    pages=loader.load()

    # Embedding and Vector Store
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
    db=FAISS.from_documents(pages,embeddings)
    retriever=db.as_retriever()

    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",google_api_key=GOOGLE_API_KEY)
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    st.success("Document uploaded and ready for questions!")
else:
    st.warning("Please upload a PDF document to get started.")

# form state
if "messages" not in st.session_state:
    st.session_state.messages=[]

if "form_state" not in st.session_state:
    st.session_state.form_state = {"step": 0, "name": "", "email": "", "phone": "", "date": ""}

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

intent_keywords=["call","appointment","schedule"]

if user_input is not None:
    intent_triggered = any(word in str(user_input).lower() for word in intent_keywords)
else:
    intent_triggered = False


#form when it is triggered
if intent_triggered or st.session_state.form_state["step"]>0:
    step=st.session_state.form_state["step"]
    if step==0:
        st.session_state.form_state["step"]+=1
        bot_msg="SURE! Enter your full name for the session"
    elif step ==1:
        st.session_state.form_state["name"]=user_input
        st.session_state.form_state["step"]+=1
        bot_msg="Enter your email"
    elif step==2:
        if re.match(r"[a-zA-Z0-9_.+-]+@[a-zA-Z]+\.[a-zA-Z]+$",user_input):
            st.session_state.form_state["email"]=user_input
            st.session_state.form_state["step"] += 1
            bot_msg = "Thanks! What's your phone number?"
        else:
            bot_msg="Please, Enter a valid email"
    elif step ==3:
        if re.match(r"^\+?\d{7,15}$",user_input):
            st.session_state.form_state["phone"] = user_input
            st.session_state.form_state["step"] += 1
            bot_msg = "And what date do you want to book? (e.g.,year-month-day OR next Monday)"
        else:
            bot_msg = "Please enter a valid phone number:"
    elif step == 4:
        parsed_date=dateparser.parse(user_input)
        if parsed_date:
                date_str = parsed_date.strftime("%Y-%m-%d")
                st.session_state.form_state["date"] = date_str
                st.session_state.form_state["step"] += 1
                # Booking confirmation
                data = st.session_state.form_state
                bot_msg = (
                    f"Appointment booked for:\n\n"
                    f"**Name:** {data['name']}\n\n"
                    f"**Email:** {data['email']}\n\n"
                    f"**Phone:** {data['phone']}\n\n"
                    f"**Date:** {data['date']}"
                )
        else:
            bot_msg = "Sorry, I couldn't understand that date. Try something like 'tomorrow' or '2025-07-15'."
    else:
            bot_msg = "Appointment already booked."
    
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

    if user_input.lower().strip()=="restart":
        st.session_state.form_state={"step": 0, "name": "", "email": "", "phone": "", "date": ""}

else:
    if user_input and user_input.strip():
        response = qa_chain.run(user_input)

    else:
        response = "Ask a question or book an appointment."

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)