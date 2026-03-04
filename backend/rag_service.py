from dotenv import load_dotenv
load_dotenv() 
import os, re
from typing import List
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# If any of these phrases are found, the query is rejected before reaching the LLM.
INJECTION_PATTERNS = [
    r"ignore (previous|above|all) instructions", r"forget (everything|all|previous)",
    r"you are now", r"act as", r"pretend (you are|to be)",
    r"disregard (previous|all|above)", r"override (instructions|prompt|system)",
    r"new instructions", r"system prompt", r"jailbreak",
]

# The instruction template sent to the LLM on every query.
PROMPT_TEMPLATE = """Answer using only the provided context.
If the question asks for a summary, generate it from the context.
If answer not found, say:
"I cannot find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:"""


# Ensures the embedding model is only loaded once across the app's lifetime.
@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def is_prompt_injection(text: str) -> bool:
    # Lowercase the input so matching is case-insensitive
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


# Custom LangChain wrapper adapting SentenceTransformer for vector store compatibility.
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = load_embedding_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Encodes a list of text chunks into vectors for indexing
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        # Encodes a single query string into a vector for similarity search
        return self.model.encode([text])[0].tolist()


class RAGService:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings()
        # Initialize the Groq LLM — API key is pulled from the environment
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0.4)
        self.vector_store = None  # Holds the in-memory Qdrant vector store after PDF is processed.
        self.chunk_count = 0      # Tracks how many chunks were created & used to cap top_k safely.

    def process_pdf(self, pdf_path: str, chunk_size: int, chunk_overlap: int):
        # Load all pages from the PDF as LangChain Document objects
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
        except Exception:
            return {"success": False, "message": "Couldn't read this PDF file."}

        # Filter out blank pages that would add noise to the index
        docs = [d for d in docs if d.page_content.strip()]
        if not docs:
            return {"success": False, "message": "No extractable text found in this PDF."}

        # Segmenting pages into overlapping chunks for more precise retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = splitter.split_documents(docs)

        if not splits:
            return {"success": False, "message": "Unable to process document text."}

        # Build an in-memory Qdrant vector store from the chunks
        # Each chunk is embedded and stored for semantic similarity search
        try:
            self.vector_store = Qdrant.from_documents(
                splits, self.embeddings, location=":memory:", collection_name="pdf_docs"
            )
        except Exception:
            return {"success": False, "message": "Failed to build search index."}

        self.chunk_count = len(splits)
        return {"success": True, "chunks": self.chunk_count, "message": f"Processed {self.chunk_count} chunks successfully."}

    def query(self, question: str, k: int, model_name: str = None):
        if not self.vector_store:
            return {"success": False, "message": "No document processed."}

        # Block any question that looks like a prompt injection attempt
        if is_prompt_injection(question):
            return {"success": False, "message": "Invalid question detected."}

        # Swap the LLM model if a different one was requested from the frontend
        if model_name and model_name != self.llm.model_name:
            self.llm = ChatGroq(model_name=model_name, api_key=os.getenv("GROQ_API_KEY"), temperature=0.4)

        # Cap k to avoid requesting more chunks than actually exist
        k = min(k, self.chunk_count)

        # Retrieve the top-k most semantically similar chunks to the question
        docs = self.vector_store.similarity_search(question, k=k)

        if not docs:
            return {"success": False, "message": "Couldn't find relevant content."}

        # Concatenate retrieved chunks into a single context block for the LLM
        context = "\n\n".join(d.page_content for d in docs if d.page_content.strip())
        if not context:
            return {"success": False, "message": "Retrieved content was empty."}

        # Build the prompt and pipe it through the LLM using LangChain's chain syntax
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": question})

        # Append the source page numbers from the retrieved chunk metadata
        answer = response.content.strip() + "\n\nSource Pages:\n"
        for doc in docs[:2]:
            page = (doc.metadata.get("page") or 0) + 1 if isinstance(doc.metadata, dict) else 1
            answer += f"  • Page {page}\n"

        return {"success": True, "answer": answer}

    def clear(self):
        # Wipe the vector store and reset chunk count on session reset
        self.vector_store = None
        self.chunk_count = 0


def check_groq():
    key = os.getenv("GROQ_API_KEY")
    return bool(key and key.startswith("gsk_"))