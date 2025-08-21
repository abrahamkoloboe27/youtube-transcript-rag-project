from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance, PayloadSchemaType
import os
from dotenv import load_dotenv
from .loggings import configure_logging
import uuid

logger = configure_logging(log_file="qdrant.log", logger_name="__qdrant__")
load_dotenv()

def get_qdrant_client() -> QdrantClient:
    """
    Returns the Qdrant client.

    Returns:
        QdrantClient: The Qdrant client.
    """
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120
    )
    logger.info(f"Connected to Qdrant. Collections: {qdrant_client.get_collections()}")
    return qdrant_client

def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE):
    """
    Creates a collection in Qdrant if it doesn't already exist and ensures required indexes are present.

    Args:
        client: The Qdrant client.
        collection_name: The name of the collection.
        vector_size: The size of the vectors.
        distance: The distance metric (e.g., COSINE, EUCLID).
    """
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    if collection_name not in collection_names:
        logger.info(f"Creating collection '{collection_name}' with vector size {vector_size} and distance {distance}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

        # Créer les index nécessaires
        _create_required_indexes(client, collection_name)
    else:
        logger.info(f"Collection '{collection_name}' already exists.")
        # Vérifier et créer les index si nécessaire (utile si la collection existait avant l'ajout de ces index)
        # Note: Qdrant ne permet pas de lister facilement les index existants, donc on tente de les créer
        # et on capture les erreurs si ils existent déjà.
        _create_required_indexes(client, collection_name)
        
        
def _create_required_indexes(client: QdrantClient, collection_name: str):
    """Helper function to create required payload indexes.
    Args:
        client: The Qdrant client.
        collection_name: The name of the collection.
        
        embedding_model_name: The name of the embedding model.
    """
    try:
        logger.info(f"Creating index on payload field 'video_id'")
        client.create_payload_index(
            collection_name=collection_name,
            field_name="video_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception as e:
        logger.debug(f"Could not create index on 'video_id' (might already exist): {e}")
    
    try:
        logger.info(f"Creating index on payload field 'embedding_model'")
        client.create_payload_index(
            collection_name=collection_name,
            field_name="embedding_model",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception as e:
        logger.debug(f"Could not create index on 'embedding_model' (might already exist): {e}")


def create_video_id_index(client: QdrantClient, collection_name: str):
    """
    Creates an index on the 'video_id' field in an existing collection.
    Should only be executed once per collection.
    """
    try:
        logger.info(f"Creating index on 'video_id' field for collection '{collection_name}'...")
        client.create_payload_index(
            collection_name=collection_name,
            field_name="video_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        logger.info("✅ Successfully created 'video_id' index.")
    except Exception as e:
        logger.error(f"❌ Error creating 'video_id' index: {e}")
        

def upsert_points(client: QdrantClient, collection_name: str, points: list[PointStruct]):
    """
    Upserts (inserts or updates) points into a Qdrant collection.

    Args:
        client: The Qdrant client.
        collection_name: The name of the collection.
        points: A list of PointStruct objects.
    """
    logger.info(f"Upserting {len(points)} points into collection '{collection_name}'")
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    logger.info(f"Successfully upserted points into collection '{collection_name}'")


def check_video_exists(client: QdrantClient, collection_name: str, video_id: str) -> bool:
    """
    Checks if a video already exists in the Qdrant collection.
    
    Args:
        client: The Qdrant client
        collection_name: Name of the collection
        video_id: YouTube video ID
        
    Returns:
        bool: True if the video exists, False otherwise
    """
    try:
        # Count points with the specific video_id
        count_result = client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id",
                        match=models.MatchValue(value=video_id)
                    )
                ]
            )
        )
        return count_result.count > 0
    except Exception as e:
        logger.error(f"Error checking existence of video {video_id}: {e}")
        return False