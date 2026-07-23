from dotenv import load_dotenv
load_dotenv()
import os, re, uuid as uuid_lib
from datetime import datetime, timezone, timedelta
from typing import List
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyMuPDFLoader  # Switched from PyPDFLoader — better table & layout extraction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,  # needed for per-user filtering in the shared collection
)

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
If the context contains table data, interpret it accurately.
If answer not found, say:
"I cannot find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:"""

# Query rewriting prompt — rephrases the user's question to be more retrieval-friendly.
# Short/vague questions often don't match chunk embeddings well; this improves recall.
REWRITE_TEMPLATE = """Rewrite the following question to be more specific and search-friendly for document retrieval.
Return only the rewritten question, nothing else.

Original question: {question}

Rewritten question:"""

CHAT_HISTORY_TTL_DAYS = 7
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# Single shared collection for all users — avoids hitting Qdrant free tier's 5-collection limit.
# Per-user isolation is handled by filtering on the user_id payload field at query time.
SHARED_COLLECTION = "querify_docs"


# Ensures the embedding model is only loaded once across the app's lifetime.
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


def _get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if url and api_key:
        print(f"[qdrant] connecting to cloud: {url}")
        return QdrantClient(url=url, api_key=api_key)
    # Local fallback for dev — data is lost on restart
    print("[qdrant] WARNING: no QDRANT_URL/QDRANT_API_KEY — using in-memory (data lost on restart)")
    return QdrantClient(":memory:")


def _user_filter(user_id: str) -> Filter:
    # Builds a Qdrant filter that scopes all searches to only this user's chunks.
    # This is the core of the single-collection multi-user isolation strategy.
    return Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    )


class RAGService:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings()
        # Initialize the Groq LLM — API key is pulled from the environment
        self.llm = ChatGroq(model_name="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)
        self.client = _get_qdrant_client()

        # In-memory chat history: { user_id: [ { question, answer, ... }, ... ] }
        # Entries are appended in order and expired lazily on read via get_history()
        self._history: dict[str, list] = {}

        # Create the shared collection once on startup if it doesn't exist yet
        self._ensure_shared_collection()

    def _ensure_shared_collection(self):
        # Called once at startup — safe to call multiple times, it's idempotent
        existing = [c.name for c in self.client.get_collections().collections]
        if SHARED_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=SHARED_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            print(f"[qdrant] created shared collection: {SHARED_COLLECTION}")
        else:
            print(f"[qdrant] shared collection already exists: {SHARED_COLLECTION}")

        # Qdrant Cloud requires an index on any payload field used in filters.
        # Creates a keyword index on user_id so count/search/delete filters work.
        # Safe to call even if the index already exists — Qdrant ignores duplicates.
        self.client.create_payload_index(
            collection_name=SHARED_COLLECTION,
            field_name="user_id",
            field_schema="keyword",
        )

    def _delete_user_points(self, user_id: str):
        # Wipes all existing vectors for this user before re-indexing a new PDF.
        # Uses a payload filter delete so other users' data is untouched.
        self.client.delete(
            collection_name=SHARED_COLLECTION,
            points_selector=_user_filter(user_id),
        )

    def process_pdf(self, pdf_path: str, chunk_size: int, chunk_overlap: int, user_id: str):
        # PyMuPDFLoader preserves layout better than PyPDFLoader — handles tables, columns, and formatting more accurately
        try:
            loader = PyMuPDFLoader(pdf_path)
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

        # Remove any previously indexed chunks for this user before inserting new ones
        self._delete_user_points(user_id)

        texts = [s.page_content for s in splits]
        vectors = self.embeddings.embed_documents(texts)

        points = []
        for i, (split, vec) in enumerate(zip(splits, vectors)):
            page = (split.metadata.get("page") or 0) + 1 if isinstance(split.metadata, dict) else 1
            points.append(PointStruct(
                # Use a UUID so point IDs are globally unique across all users in the shared collection.
                # Sequential ints would collide between users.
                id=str(uuid_lib.uuid4()),
                vector=vec,
                payload={
                    "text": split.page_content,
                    "page": page,
                    "user_id": user_id,  # stored in payload so we can filter by it at search time
                }
            ))

        # Upsert in batches of 100 to stay within free tier request limits
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=SHARED_COLLECTION, points=points[i:i + batch_size])

        chunk_count = len(splits)
        return {"success": True, "chunks": chunk_count, "message": f"Processed {chunk_count} chunks successfully."}

    def rewrite_query(self, question: str) -> str:
        # Ask the LLM to rephrase the question for better embedding match — runs fast on 8B
        try:
            prompt = PromptTemplate(template=REWRITE_TEMPLATE, input_variables=["question"])
            chain = prompt | self.llm
            result = chain.invoke({"question": question})
            rewritten = result.content.strip()
            # Fall back to the original if the rewrite came back empty
            return rewritten if rewritten else question
        except Exception:
            return question

    def query(self, question: str, k: int, user_id: str, model_name: str = None):
        # Check if this user has any indexed chunks before hitting the LLM
        if not self.has_document(user_id):
            return {"success": False, "message": "No document processed."}

        # Block any question that looks like a prompt injection attempt
        if is_prompt_injection(question):
            return {"success": False, "message": "Invalid question detected."}

        # Swap the LLM model if a different one was requested from the frontend
        if model_name and model_name != self.llm.model_name:
            self.llm = ChatGroq(model_name=model_name, api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)

        # Rewrite the query before retrieval to improve chunk matching
        rewritten_question = self.rewrite_query(question)

        # Retrieve the top-k most semantically similar chunks scoped to this user only
        query_vec = self.embeddings.embed_query(rewritten_question)
        results = self.client.search(
            collection_name=SHARED_COLLECTION,
            query_vector=query_vec,
            limit=k,
            query_filter=_user_filter(user_id),  # keeps users isolated
        )

        if not results:
            return {"success": False, "message": "Couldn't find relevant content."}

        # Concatenate retrieved chunks into a single context block for the LLM
        context_parts = [r.payload["text"] for r in results if r.payload.get("text", "").strip()]
        context = "\n\n".join(context_parts)
        if not context:
            return {"success": False, "message": "Retrieved content was empty."}

        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": rewritten_question})

        # Build structured citations
        citations = []
        seen_pages = set()
        for r in results:
            page = r.payload.get("page", 1)
            if page not in seen_pages:
                seen_pages.add(page)
                citations.append({
                    "page": page,
                    "snippet": r.payload.get("text", "")[:300]  # First 300 chars as a preview
                })

        answer = response.content.strip()

        # Persist this Q&A turn to in-memory history
        self._save_history_entry(user_id, question, answer, rewritten_question, citations)

        return {
            "success": True,
            "answer": answer,
            "citations": citations,
            "rewritten_question": rewritten_question,  # Sent back so the debug panel can show it
            "chunks_used": len(results),
            "chunks_retrieved": [r.payload.get("text", "")[:200] for r in results],  # Short previews for debug panel
        }

    def _save_history_entry(self, user_id: str, question: str, answer: str,
                            rewritten_question: str, citations: list):
        now = datetime.now(timezone.utc)
        entry = {
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "rewritten_question": rewritten_question,
            "citations": citations,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=CHAT_HISTORY_TTL_DAYS)).isoformat(),
        }
        # Initialise the list for this user if it doesn't exist yet
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(entry)

    def get_history(self, user_id: str) -> list:
        now_iso = datetime.now(timezone.utc).isoformat()
        entries = self._history.get(user_id, [])
        valid = [e for e in entries if e.get("expires_at", "9999") > now_iso]
        self._history[user_id] = valid
        return valid

    def clear(self, user_id: str):
        self._delete_user_points(user_id)
        self._history.pop(user_id, None)

    def has_document(self, user_id: str) -> bool:
        # Count vectors in the shared collection that belong to this user
        result = self.client.count(
            collection_name=SHARED_COLLECTION,
            count_filter=_user_filter(user_id),
            exact=True,
        )
        return result.count > 0

def check_groq():
    key = os.getenv("GROQ_API_KEY")
    return bool(key and key.startswith("gsk_"))
