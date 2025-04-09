# streamlit_app.py
import streamlit as st
import os
import time
from document_assistant import DocumentAssistant

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "doc_assistant" not in st.session_state:
        st.session_state.doc_assistant = None
    
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    
    if "document_summary" not in st.session_state:
        st.session_state.document_summary = ""

def display_chat_messages():
    """Display chat messages from session state"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def add_message(role, content):
    """Add a message to the chat history"""
    st.session_state.messages.append({"role": role, "content": content})

def clear_chat_history():
    """Clear the chat history"""
    st.session_state.messages = []

def main():
    st.set_page_config(
        page_title="Document Assistant Chat",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Assistant")
       
    init_session_state()
        
    with st.sidebar:
        st.header("Document Upload")
        
    
        uploaded_file = st.file_uploader("Upload a PDF contract", type="pdf")
        
    
        st.header("Configuration")
        chunk_strategy = st.selectbox(
            "Chunking Strategy",
            ["recursive", "semantic"],
            help="Choose how to split the document into chunks"
        )
        
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["similarity", "hybrid"],
            help="Choose how to retrieve relevant chunks for answering questions"
        )
        
    
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing document..."):
    
                st.session_state.doc_assistant = DocumentAssistant(chunk_strategy, retrieval_method)
                
    
                text = st.session_state.doc_assistant.extract_text_from_pdf(uploaded_file)
                
    
                st.session_state.doc_assistant.process_document(text)
                
    
                st.session_state.document_summary = st.session_state.doc_assistant.get_summary()
                
    
                st.session_state.document_processed = True
                
    
                system_message = f"Document processed successfully! Ask me questions about the document or request a summary."
                add_message("assistant", system_message)
        
    
        if st.button("Clear Chat History"):
            clear_chat_history()
            st.success("Chat history cleared!")
        
        st.divider()
        st.write("This application was developed for an interview with CriticalRiver.")
    
    
    tab1, tab2 = st.tabs(["Chat", "Document Summary"])
    
    # Chat tab
    with tab1:
        # Display chat messages
        display_chat_messages()
        
         
        if not st.session_state.document_processed:
            st.info("Please upload and process a document first using the sidebar.")
        else:
         
            if prompt := st.chat_input("Ask a question about the document..."):
         
                add_message("user", prompt)
                
        
                with st.chat_message("user"):
                    st.markdown(prompt)
                
        
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.doc_assistant.ask_question(prompt)
                        st.markdown(response)
                
        
                add_message("assistant", response)
            
        
            st.subheader("Sample Questions")
            col1, col2 = st.columns(2)
            
            sample_questions = [
                "What are the main parties in this contract?",               
                "What are the payment terms?",
                "What is the duration of this contract?",
                "What are the termination conditions?"
            ]
            
            
            with col1:
                for q in sample_questions[:3]:
                    if st.button(q):
                       
                        add_message("user", q)
                        
                       
                        with st.spinner("Thinking..."):
                            response = st.session_state.doc_assistant.ask_question(q)
                        
                       
                        add_message("assistant", response)
                        
                       
                        st.experimental_rerun()
            
            
            with col2:
                for q in sample_questions[3:]:
                    if st.button(q):
            
                        add_message("user", q)
                                                
                        with st.spinner("Thinking..."):
                            response = st.session_state.doc_assistant.ask_question(q)
                                                
                        add_message("assistant", response)
                                            
                        st.experimental_rerun()
    
    
    with tab2:
        if st.session_state.document_processed:
    
            st.subheader("Document Metrics")
            metrics = st.session_state.doc_assistant.get_metrics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Document Length", metrics.get("Document Length", 0))
                st.metric("Chunking Strategy", metrics.get("Chunking Strategy", "N/A"))
            with col2:
                st.metric("Word Count", metrics.get("Word Count", 0))
                st.metric("Retrieval Method", metrics.get("Retrieval Method", "N/A"))
            
            st.subheader("Document Summary")
            st.write(st.session_state.document_summary)
                        
            if st.button("Regenerate Summary"):
                with st.spinner("Generating new summary..."):
                    st.session_state.document_summary = st.session_state.doc_assistant.get_summary()
                    st.success("Summary regenerated!")
                    st.experimental_rerun()
        else:
            st.info("Please upload and process a document first using the sidebar.")

if __name__ == "__main__":
    main()
