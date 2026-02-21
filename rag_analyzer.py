import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import streamlit as st

load_dotenv()


# Cache the embedding model so it doesn't reload on every Streamlit rerun
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = load_embedding_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()


class RAGAnalyzer:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.embeddings = SentenceTransformerEmbeddings()

        self.llm = ChatGroq(
            model_name=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.6
        )

        self.vector_store = None
        self.chunk_count = 0

    def process_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except Exception:
            return {"success": False, "message": "Couldn't read this PDF file."}

        if not documents:
            return {"success": False, "message": "The PDF has no readable pages."}

        # Remove blank pages to avoid indexing empty content
        documents = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        if not documents:
            return {"success": False, "message": "No extractable text found in this PDF."}

        # Split long text into smaller overlapping chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            return {"success": False, "message": "Unable to process document text."}

        try:
            self.vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                location=":memory:",
                collection_name="pdf_docs"
            )
        except Exception:
            return {"success": False, "message": "Failed to build search index."}

        # Track number of chunks to prevent top_k overflow
        self.chunk_count = len(chunks)

        return {
            "success": True,
            "message": f"Processed {self.chunk_count} chunks successfully."
        }

    def answer_question(self, question: str, k: int = 4):
        if self.vector_store is None:
            return {"success": False, "message": "Upload and process a PDF first."}

        # Constrain chunk selection to valid range
        k = min(k, self.chunk_count)

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            return {"success": False, "message": "Couldn't find relevant content."}

        # Combine retrieved chunks into a single context block
        context = "\n\n".join(
            doc.page_content
            for doc in relevant_docs
            if doc.page_content.strip()
        )

        if not context:
            return {"success": False, "message": "Retrieved content was empty."}

        prompt = PromptTemplate(
            template="""Answer the question using the provided context.
For analytical or evaluative questions (e.g., rating, feedback, suggestions),
base your reasoning strictly on the retrieved content.
Do not introduce external information.

Context:
{context}

Question:
{question}

Answer:""",
            input_variables=["context", "question"]
        )

        response = self.llm.invoke(
            prompt.format(context=context, question=question)
        )

        answer_text = response.content.strip()
        answer_text += "\n\nSource Pages:\n"

        # extract page numbers from metadata
        for doc in relevant_docs[:2]:
            page_number = 1
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                page_number = (doc.metadata.get("page") or 0) + 1
            answer_text += f"  • Page {page_number}\n"

        return {"success": True, "answer": answer_text}

    def clear_vector_store(self):
        self.vector_store = None
        self.chunk_count = 0


def check_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.startswith("gsk_"):
        return True, [
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
    return False, []