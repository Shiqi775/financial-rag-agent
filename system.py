import os
import torch
from typing import Optional, List, Tuple
from llama_index.core import QueryBundle, VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters
import chromadb
import time
import together

from config import VECTOR_STORE_DIR, EMBED_MODEL_NAME, LLM_MODEL_NAME, TOGETHER_API_KEY

RETRIEVER_SIMILARITY_TOP_K: int = 5
RERANKER_CHOICE_BATCH_SIZE: int = 3
RERANKER_TOP_N: int = 1


class RAGSystem:
    """RAG system for SEC 10-K financial document retrieval and question answering."""

    def __init__(self):
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize embedding model
        try:
            self._embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        except RuntimeError as e:
            print(f"Error loading embedding model on GPU: {e}")
            print("Falling back to CPU...")
            self.device = "cpu"
            self._embed_model = HuggingFaceEmbedding(
                model_name=EMBED_MODEL_NAME,
                device="cpu",
            )

        # Initialize vector database from existing collection
        self._chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        try:
            self._collection = self._chroma_client.get_collection("financial_filings")
        except Exception:
            self._collection = self._chroma_client.get_or_create_collection(
                "financial_filings"
            )

        # Initialize vector store and index
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._vector_index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model
        )

        # Initialize Together.ai API for LLM (optional, for legacy support)
        if TOGETHER_API_KEY:
            together.api_key = TOGETHER_API_KEY
        else:
            print("Warning: TOGETHER_API_KEY not set. Legacy LLM features will not work.")

        # Initialize reranker
        self._reranker = SimilarityPostprocessor(
            choice_batch_size=RERANKER_CHOICE_BATCH_SIZE,
            top_n=RERANKER_TOP_N,
        )

        # Cache available tickers
        self._available_tickers = self._load_available_tickers()

    def _load_available_tickers(self) -> List[str]:
        """Load available tickers from file."""
        try:
            ticker_file = "sampled_tickers.txt"
            if os.path.exists(ticker_file):
                with open(ticker_file, "r") as file:
                    return [line.strip() for line in file.readlines()]
        except Exception:
            pass
        return []

    def get_available_tickers(self) -> List[str]:
        """Return list of available company tickers."""
        return self._available_tickers

    def retrieve(self, query: str, ticker: str, year: Optional[str]) -> List[TextNode]:
        """
        Retrieve relevant document nodes with graceful fallback on year.
        """
        query_bundle = QueryBundle(f"{ticker} {query}")

        metadata_filters = MetadataFilters(
            filters=[
                {"key": "ticker", "value": ticker},
                {"key": "year", "value": year}
            ]
        )
        
        retriever = self._vector_index.as_retriever(
            similarity_top_k=RETRIEVER_SIMILARITY_TOP_K,  # 5
            filters=metadata_filters
        )
        nodes = retriever.retrieve(query_bundle)

        filtered_nodes = [
            node for node in nodes
            if node.metadata.get("ticker") == ticker and
               node.metadata.get("year") == year
        ]

        self._reranker = SimilarityPostprocessor(
            choice_batch_size=RERANKER_CHOICE_BATCH_SIZE,
            top_n=RERANKER_TOP_N,
        )
        reranked_nodes = self._reranker.postprocess_nodes(filtered_nodes, query_bundle)
        return reranked_nodes[:2]  # Return top 2 nodes
        # # ---------- 1. Try ticker + year ----------
        # nodes: List[TextNode] = []
        # if year:
        #     strict_filters = MetadataFilters(
        #         filters=[
        #             {"key": "ticker", "value": ticker},
        #             {"key": "year", "value": year},
        #         ]
        #     )

        #     retriever = self._vector_index.as_retriever(
        #         similarity_top_k=RETRIEVER_SIMILARITY_TOP_K,
        #         filters=strict_filters,
        #     )
        #     nodes = retriever.retrieve(query_bundle)

        # # ---------- 2. fallback: ticker only ----------
        # if not nodes:
        #     relaxed_filters = MetadataFilters(
        #         filters=[{"key": "ticker", "value": ticker}]
        #     )

        #     retriever = self._vector_index.as_retriever(
        #         similarity_top_k=RETRIEVER_SIMILARITY_TOP_K,
        #         filters=relaxed_filters,
        #     )
        #     nodes = retriever.retrieve(query_bundle)

        # if not nodes:
        #     return []

        # # ---------- 3. rerank ----------
        # reranked_nodes = self._reranker.postprocess_nodes(nodes, query_bundle)
        # return reranked_nodes[:2] if reranked_nodes else []


    def retrieve_and_respond(self, query: str, ticker: str, year: str) -> Tuple[str, Optional[List[TextNode]]]:
        """
        Retrieve documents and generate a response (used by RAG tool).

        Args:
            query: The user's question
            ticker: Company ticker symbol
            year: Year to query

        Returns:
            Tuple of (response string, list of source nodes)
        """
        nodes = self.retrieve(query, ticker, year)

        if not nodes:
            return "No relevant information found.", None


        # Combine node texts for context
        context_text = " ".join(node.text for node in nodes if node.text)

        # Generate summary response from context without LLM
        # (The agent will use this context to formulate its own response)
        snippet = context_text[:1500] if len(context_text) > 1500 else context_text
        return f"Found relevant information: {snippet}", nodes

    def respond(self, query: str, ticker: str, year: str) -> Tuple[str, Optional[List[TextNode]]]:
        """
        Generate a response using Together.ai LLM.

        Args:
            query: The user's question
            ticker: Company ticker symbol
            year: Year to query

        Returns:
            Tuple of (generated response, list of source nodes)
        """
        nodes = self.retrieve(query, ticker, year)

        if not nodes:
            return "No relevant information found for this company and year.", None

        # Combine node texts for context
        context_text = " ".join(node.text for node in nodes if node.text)

        prompt = (
            f"Based on the following context, answer the question: {query}\n\n"
            f"Context from {ticker}'s {year} report:\n{context_text}\n\n"
            "Answer the question concisely in one sentence:"
        )

        for attempt in range(5):  # Retry up to 5 times
            try:
                response = together.Completion.create(
                    model=LLM_MODEL_NAME,
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["\n"]
                )
                response_text = response.choices[0].text.strip()
                return response_text, nodes
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}", flush=True)
                time.sleep(2 ** attempt)  # Exponential backoff

        return "LLM API failed after multiple attempts.", []


if __name__ == '__main__':
    # Simple test
    system = RAGSystem()
    print(f"Available tickers: {system.get_available_tickers()}")
    print("RAG system initialized successfully.")
