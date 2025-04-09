# document_assistant.py
import os
import time
import torch
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tempfile
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")  
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
HF_API_KEY = os.getenv("HF_API_KEY", "")  

class DocumentAssistant:
    def __init__(self, chunk_strategy="recursive", retrieval_method="similarity"):
        self.chunk_strategy = chunk_strategy
        self.retrieval_method = retrieval_method
        self.document_text = ""
        self.vectorstore = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = None
            
        self.embeddings = self._load_embeddings()
        
        self.llm = self._load_llm()
        
        logger.info(f"Initialized DocumentAssistant with chunk_strategy={chunk_strategy} and retrieval_method={retrieval_method}")

    def _load_embeddings(self):
        """Load the embeddings model"""
        logger.info(f"Loading embeddings model: {EMBEDDINGS_MODEL}")
                
        if HF_API_KEY:
            return HuggingFaceEmbeddings(
                model_name=EMBEDDINGS_MODEL,        
            )
        else:
            return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    def _load_llm(self):
        """Load the language model for Q&A and summarization"""
        logger.info(f"Loading LLM model: {LLM_MODEL}")
        
        # Check if API key is provided
        auth_token = HF_API_KEY if HF_API_KEY else None
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL,
                token=auth_token
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                token=auth_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        
        except Exception as e:
            logger.error(f"Error loading model {LLM_MODEL}: {str(e)}")
            logger.info("Falling back to default model")
            
            # Fallback to a simpler model if the specified one fails
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            model = AutoModelForCausalLM.from_pretrained(
                "google/flan-t5-small",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE
            )
            
            return HuggingFacePipeline(pipeline=pipe)

    def extract_text_from_pdf(self, pdf_file):
        logger.info("Extracting text from PDF")
        
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name
        
        # Extract text using PyPDF2
        pdf_reader = PdfReader(temp_file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        self.document_text = text
        return text

    def process_document(self, text):
        logger.info(f"Processing document with chunk strategy: {self.chunk_strategy}")
        
        # Create text chunks based on selected strategy
        if self.chunk_strategy == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
        else:  # semantic chunking
            try:
                text_splitter = SpacyTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
            except:
                # Fallback to recursive if SpaCy is not available
                logger.warning("SpaCy not available, falling back to recursive chunking")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len
                )
        
        chunks = text_splitter.split_text(text)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        
        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)
        
       
        if self.retrieval_method == "similarity":
            retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        else:  
            retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
        
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory
        )

    def get_summary(self):
        
        logger.info("Generating document summary")        
        if not self.document_text:
            return ""
        
        prompt = f"""
        Please provide a concise summary of the following document in about 100 words:
        
        {self.document_text}  
        
        Summary:
        """
        
        return self.llm.invoke(prompt)

    def ask_question(self, question):

        logger.info(f"Processing question: {question}")        
        if not self.qa_chain:
            return "Please upload and process a document first."    

        if question.lower().startswith("follow up"):

            actual_question = question.split(":", 1)[1].strip()        
        
            if "Standalone question:" in actual_question:
                parts = actual_question.split("Standalone question:")
                actual_question = parts[1].strip()
        else:
            actual_question = question

        start_time = time.time()
        try:
            response = self.qa_chain({"question": actual_question})
            answer = response['answer'].strip()
            
            if "to answer the question at the end." in answer:
                answer = answer.split("to answer the question at the end.")[-1]
            
            answer = answer.split("Question:")[-1]  
            if "Helpful Answer:" in answer:
                answer = "Helpful Answer:" + answer.split("Helpful Answer:")[-1]
            else:
                answer = "Helpful Answer: " + answer
                
            end_time = time.time()
            logger.info(f"Question answered in {end_time - start_time:.2f} seconds")
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return "Sorry, I encountered an error while processing your question."
   
    def get_metrics(self):
        if not self.document_text:
            return {}
        logger.info("Collecting document metrics")
        return {
            "Document Length": len(self.document_text),
            "Word Count": len(self.document_text.split()),
            "Chunking Strategy": self.chunk_strategy,
            "Retrieval Method": self.retrieval_method,
            "LLM Model": LLM_MODEL,
            "Embeddings Model": EMBEDDINGS_MODEL
        }