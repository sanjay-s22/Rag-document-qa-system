import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

class RAGAnalyzer:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            model_name=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7
        )
        self.vector_store = None
        self.pdf_name = None

    def process_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            if not documents:
                return "Error: PDF is empty"

            self.pdf_name = os.path.basename(pdf_path)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)

            self.vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                location=":memory:",
                collection_name="pdf_docs"
            )
            return f"✅ Successfully processed PDF! Created {len(chunks)} chunks"

        except Exception as e:
            return f"Error: {str(e)}"

    def answer_question(self, question: str, k: int = 4) -> str:
        if self.vector_store is None:
            return "Please upload a PDF first"

        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            relevant_docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt_template = """You are a helpful assistant analyzing a document.
Use the context to answer. If asked to rate/analyze/evaluate,
use the information in the context to give your best assessment.

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
                    page_num = doc.metadata.get("page", "Unknown")
                    response += f"  • Page {page_num}\n"
            return response

        except Exception as e:
            return f"Error: {str(e)}"

    def clear_vector_store(self):
        self.vector_store = None
        self.pdf_name = None


def check_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.startswith("gsk_"):
        return True, ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
    return False, []