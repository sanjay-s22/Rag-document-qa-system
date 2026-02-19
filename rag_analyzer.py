import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from typing import List

load_dotenv()

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

class RAGAnalyzer:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            model_name=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.6 # can adjust from 0 to 1, lower = more focused, higher = more creative 
        )
        self.vector_store = None
        self.pdf_name = None

    def process_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> dict:
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

        
            if not documents:
                return {
                    "success": False,
                    "message": "PDF contains no extractable text."
                }

            self.pdf_name = os.path.basename(pdf_path)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]    # Padding to keep sentences whole; don't want to cut a thought in half mid-chunk
            )
            chunks = text_splitter.split_documents(documents)

            self.vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                location=":memory:",
                collection_name="pdf_docs"
            )
            return {
                "success": True,
                "chunks": len(chunks),
                "message": f"Processed {len(chunks)} text chunks."
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    def answer_question(self, question: str, k: int = 4) -> dict:  
        # Safety check to ensure the vector index exists before we attempt a retrieval call
        if self.vector_store is None:
            return {
            "success": False,
            "message": "Please upload a PDF first"
        }

        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            relevant_docs = retriever.invoke(question)                           # always returns top-k closest chunks
            context = "\n\n".join([doc.page_content for doc in relevant_docs])   # Building the prompt context by joining top-k document fragments

            prompt_template = """Answer the question using the provided context.

For analytical or evaluative questions (e.g., rating, feedback, suggestions),
base your reasoning strictly on the retrieved content.

Do not introduce external information.

Context:
{context}

Question: {question}

Answer:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            formatted_prompt = prompt.format(context=context, question=question)

            response_msg = self.llm.invoke(formatted_prompt)
            answer = response_msg.content

            response = f"{answer}\n\n"
            if relevant_docs:
                response += "Source Pages:\n"
                for doc in relevant_docs[:2]:
                    page_num = doc.metadata.get("page", 0) + 1
                    response += f"  • Page {page_num}\n"         # Limiting reference metadata to the top 2 most relevant chunks for better readability
            
            return {
                "success": True,
                "answer": response
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    def clear_vector_store(self):
        self.vector_store = None
        self.pdf_name = None


def check_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.startswith("gsk_"):
        return True, ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
    return False, []