# test-document-assistant.py
import unittest
import os
from document_assistant import DocumentAssistant
import logging
from unittest.mock import MagicMock, patch

class TestDocumentAssistant(unittest.TestCase):
    @patch('document_assistant.HuggingFaceEmbeddings')
    @patch('document_assistant.AutoTokenizer')
    @patch('document_assistant.AutoModelForCausalLM')
    @patch('document_assistant.pipeline')
    @patch('document_assistant.HuggingFacePipeline')
    def setUp(self, mock_hf_pipeline, mock_pipeline, mock_model, mock_tokenizer, mock_embeddings):
        """Set up test fixtures with mocked components"""
        # Mock embeddings
        self.mock_embeddings = mock_embeddings.return_value
        self.mock_embeddings.embed_documents = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        self.mock_embeddings.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])

        # Mock tokenizer
        self.mock_tokenizer = mock_tokenizer.from_pretrained.return_value

        # Mock model
        self.mock_model = mock_model.from_pretrained.return_value

        # Mock pipeline
        self.mock_pipeline = mock_pipeline.return_value

        # Mock HuggingFacePipeline
        self.mock_hf_pipeline = mock_hf_pipeline.return_value
        self.mock_hf_pipeline.invoke = MagicMock(return_value="This is a mocked response")

        # Create document assistant
        self.doc_assistant = DocumentAssistant()
        self.test_pdf_path = os.path.join(os.path.dirname(__file__), "assignment_ml_engineer.pdf")
        self.sample_text = "This is a sample document text for testing purposes."

    @patch('document_assistant.PdfReader')
    def test_pdf_extraction(self, mock_pdf_reader):
        """Test PDF text extraction with mocked PDF reader"""
        # Mock PDF reader behavior
        mock_page = MagicMock()
        mock_page.extract_text.return_value = self.sample_text
        mock_pdf_reader.return_value.pages = [mock_page]

        with open(self.test_pdf_path, 'rb') as pdf_file:
            result = self.doc_assistant.extract_text_from_pdf(pdf_file)
            
            self.assertEqual(result, self.sample_text)
            self.assertEqual(self.doc_assistant.document_text, self.sample_text)
            mock_pdf_reader.assert_called_once()
            mock_page.extract_text.assert_called_once()

    @patch('document_assistant.FAISS')
    @patch('document_assistant.ConversationalRetrievalChain')
    def test_document_processing(self, mock_chain, mock_faiss):
        """Test document processing with mocked vector store"""
        # Mock FAISS behavior
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss.from_texts.return_value = mock_vectorstore

        # Mock chain
        mock_chain_instance = MagicMock()
        mock_chain.from_llm.return_value = mock_chain_instance

        # Test recursive chunking
        self.doc_assistant.chunk_strategy = "recursive"
        self.doc_assistant.process_document(self.sample_text)
        
        mock_faiss.from_texts.assert_called()
        self.assertIsNotNone(self.doc_assistant.vectorstore)
        self.assertIsNotNone(self.doc_assistant.qa_chain)

        # Test semantic chunking
        self.doc_assistant.chunk_strategy = "semantic"
        self.doc_assistant.process_document(self.sample_text)
        
        mock_faiss.from_texts.assert_called()
        self.assertIsNotNone(self.doc_assistant.vectorstore)

    def test_question_answering(self):
        """Test question answering with mocked QA chain"""
        # Setup QA chain
        mock_qa_chain = MagicMock()
        mock_qa_chain.return_value = {"answer": "The duration is 4 days"}
        self.doc_assistant.qa_chain = mock_qa_chain
        self.doc_assistant.document_text = self.sample_text

        # Test basic question
        question = "What is the duration?"
        response = self.doc_assistant.ask_question(question)
        self.assertIn("Helpful Answer:", response)
        self.assertIn("4 days", response)

        # Test follow-up question
        follow_up = "Follow up: What are the requirements?"
        mock_qa_chain.return_value = {"answer": "The requirements include GitHub repository"}
        response = self.doc_assistant.ask_question(follow_up)
        self.assertIn("Helpful Answer:", response)
        self.assertIn("GitHub repository", response)

    def test_summary_generation(self):
        """Test document summarization with mocked LLM"""
        self.doc_assistant.document_text = self.sample_text
        self.doc_assistant.llm.invoke = MagicMock(return_value="This is a mocked summary")
        
        summary = self.doc_assistant.get_summary()
        self.assertEqual(summary, "This is a mocked summary")

    def test_metrics(self):
        """Test metrics collection"""
        test_text = "This is a test document with some words."
        self.doc_assistant.document_text = test_text
        
        metrics = self.doc_assistant.get_metrics()
        
        self.assertEqual(metrics["Document Length"], len(test_text))
        self.assertEqual(metrics["Word Count"], len(test_text.split()))
        self.assertEqual(metrics["Chunking Strategy"], self.doc_assistant.chunk_strategy)
        self.assertEqual(metrics["Retrieval Method"], self.doc_assistant.retrieval_method)

    def test_error_handling(self):
        """Test error handling scenarios"""
        # Test asking question without processing document
        self.doc_assistant.qa_chain = None
        response = self.doc_assistant.ask_question("What is the assignment?")
        self.assertEqual(response, "Please upload and process a document first.")

        # Test PDF extraction error
        with patch('document_assistant.PdfReader', side_effect=Exception("PDF Error")):
            with self.assertRaises(Exception):
                with open(self.test_pdf_path, 'rb') as pdf_file:
                    self.doc_assistant.extract_text_from_pdf(pdf_file)

if __name__ == '__main__':
    unittest.main()