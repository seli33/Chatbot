
# Chatbot for PDF Q&A and Appointment Booking

## Overview

This project is a chatbot built with LangChain, Google Gemini (LLM), and Streamlit that can:

- Answer questions based on the content of an uploaded PDF document.
- Collect user information (name, email, phone, date) through a conversational form for booking appointments.
- Validate user inputs including emails, phone numbers.

The chatbot uses LangChain agents and tools to dynamically route user queries to either document question answering or the appointment booking flow.

## Features

- Upload PDF documents for question answering.
- Conversational memory to maintain context.
- Powered by Google Gemini LLM and FAISS vector search.

## Technologies Used

- Python, Streamlit (UI)
- LangChain (agents, tools, memory, retrieval)
- Google Gemini via `langchain_google_genai`
- PyPDFLoader for PDF processing
- FAISS vector store for similarity search
- Dateparser for natural language date parsing
- Regex for input validation


