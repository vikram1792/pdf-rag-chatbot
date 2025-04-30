# app.py

import streamlit as st
from helper import load_and_split_pdf, setup_vectorstore, build_chain

st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")

st.title("📄 PDF Summary & Q&A Assistant")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        
        chunks = load_and_split_pdf("temp.pdf")
        vectorstore = setup_vectorstore(chunks)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        chain = build_chain(retriever)
    
    st.success("PDF processed! Ask questions below.")

    user_query = st.text_input("Ask a question about the PDF:")
    if user_query:
        with st.spinner("Generating answer..."):
            response = chain.invoke(user_query)
            st.write("**Answer:**", response)
