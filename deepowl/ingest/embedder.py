import chromadb

from deepowl.ingest.chunker import Chunk
from deepowl.llm import get_embedding

COLLECTION_NAME = "deepowl"


def get_collection(chroma_path: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_chunks(
    chunks: list[Chunk],
    collection: chromadb.Collection,
    provider: str,
    model: str,
) -> list[str]:
    """Embed chunks and upsert into ChromaDB. Returns list of IDs."""
    if not chunks:
        return []

    ids = [f"{c.source}::chunk_{c.chunk_index}" for c in chunks]
    texts = [c.content for c in chunks]
    metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]

    embeddings = [get_embedding(provider, model, text) for text in texts]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return ids


def delete_file_embeddings(collection: chromadb.Collection, source: str) -> None:
    results = collection.get(where={"source": source})
    if results["ids"]:
        collection.delete(ids=results["ids"])


def search(collection: chromadb.Collection, query_embedding: list[float], n: int = 5) -> list[dict]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "score": 1 - results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]
