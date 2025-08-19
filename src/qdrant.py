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
    Creates a collection in Qdrant if it doesn't already exist.

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

        # Create index on video_id field for fast filtering
        logger.info(f"Creating index on payload field 'video_id'")
        client.create_payload_index(
            collection_name=collection_name,
            field_name="video_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        # Create index on language field
        logger.info(f"Creating index on payload field 'language'")
        client.create_payload_index(
            collection_name=collection_name,
            field_name="language",
            field_schema=PayloadSchemaType.KEYWORD
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists.")


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