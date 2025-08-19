# embedding.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # Pour découper le texte
from .loggings import configure_logging
from .qdrant import get_qdrant_client, create_collection_if_not_exists, upsert_points
from qdrant_client.models import PointStruct
import uuid
from typing import List, Tuple
import os

logger = configure_logging(log_file="embedding.log", logger_name="__embedding__")

# Constante pour la taille du modèle d'embedding (à ajuster selon le modèle choisi)
# Pour 'sentence-transformers/all-mpnet-base-v2', la dimension est 768
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Returns an instance of the HuggingFaceEmbeddings class.

    Args:
        model_name: The name of the model to use.

    Returns:
        An instance of the HuggingFaceEmbeddings class.
    """
    logger.info(f"Loading model {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    logger.info(f"Model {model_name} loaded")
    return embeddings

def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Splits a text into chunks using RecursiveCharacterTextSplitter.

    Args:
        text: The text to split.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The overlap between chunks.

    Returns:
        A list of text chunks.
    """
    logger.info(f"Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([text]) # create_documents attend une liste
    # Extraire le texte de chaque Document
    chunk_texts = [doc.page_content for doc in chunks]
    logger.info(f"Text split into {len(chunk_texts)} chunks")
    return chunk_texts

def embed_text_chunks(text_chunks: List[str], model: HuggingFaceEmbeddings) -> List[List[float]]:
    """
    Embeds a list of text chunks.

    Args:
        text_chunks: A list of text strings to embed.
        model: The embedding model instance.

    Returns:
        A list of embedding vectors.
    """
    logger.info(f"Embedding {len(text_chunks)} text chunks")
    # embed_documents est plus efficace pour une liste de textes
    embeddings = model.embed_documents(text_chunks)
    logger.info(f"Embedded {len(embeddings)} text chunks")
    return embeddings

def process_and_store_transcript_txt(
    txt_file_path: str,
    collection_name: str,
    video_id: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100
):
    """
    Processes a TXT transcript file: loads, splits, embeds, and stores in Qdrant.

    Args:
        txt_file_path: Path to the TXT file.
        collection_name: Name of the Qdrant collection.
        video_id: The YouTube video ID (used for payload).
        chunk_size: Size of text chunks.
        chunk_overlap: Overlap between chunks.
    """
    logger.info(f"Starting processing for TXT file: {txt_file_path}")

    # 1. Charger le texte
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        logger.info(f"Loaded text from {txt_file_path}, length: {len(full_text)} chars")
    except FileNotFoundError:
        logger.error(f"File not found: {txt_file_path}")
        return
    except Exception as e:
        logger.error(f"Error reading file {txt_file_path}: {e}")
        return

    # 2. Découper le texte
    text_chunks = split_text_into_chunks(full_text, chunk_size, chunk_overlap)

    # 3. Charger le modèle d'embedding
    embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)

    # 4. Embedder les morceaux
    embeddings = embed_text_chunks(text_chunks, embedding_model)

    # 5. Préparer les points pour Qdrant
    points = []
    for i, (chunk, vector) in enumerate(zip(text_chunks, embeddings)):
        # Créer un ID unique pour chaque point
        point_id = str(uuid.uuid4()) # Ou utiliser video_id + index si préférable
        payload = {
            "video_id": video_id,
            "chunk_index": i,
            "text": chunk # Optionnel: stocker le texte brut
            # Ajouter d'autres métadonnées si nécessaire (titre, timestamp, etc.)
        }
        point = PointStruct(id=point_id, vector=vector, payload=payload)
        points.append(point)

    # 6. Se connecter à Qdrant
    qdrant_client = get_qdrant_client()

    # 7. Créer la collection si elle n'existe pas
    # On suppose que la dimension est connue (768 pour all-mpnet-base-v2)
    create_collection_if_not_exists(qdrant_client, collection_name, EMBEDDING_DIMENSION)

    # 8. Insérer les points dans Qdrant
    upsert_points(qdrant_client, collection_name, points)

    logger.info(f"Finished processing and storing {len(points)} chunks from {txt_file_path}")
