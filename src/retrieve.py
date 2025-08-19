# retrieve.py

from src.loggings import configure_logging
from src.embedding import get_embedding_model
from src.qdrant import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import List, Dict, Optional

logger = configure_logging(log_file="retrieve.log", logger_name="__retrieve__")

# Nom de la collection (doit être le même que dans main.py)
DEFAULT_COLLECTION_NAME = "youtube_transcripts"

def retrieve_relevant_chunks(
    query: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    video_id: Optional[str] = None,
    language_code: Optional[str] = None,
    top_k: int = 5
) -> List[Dict]:
    """
    Recherche les chunks pertinents dans Qdrant en fonction d'une requête.

    Args:
        query (str): La question de l'utilisateur.
        collection_name (str): Nom de la collection Qdrant.
        video_id (Optional[str]): Si spécifié, filtre les résultats par vidéo.
        language_code (Optional[str]): Si spécifié, filtre par langue.
        top_k (int): Nombre de résultats à retourner.

    Returns:
        List[Dict]: Liste des chunks pertinents avec leurs métadonnées.
    """
    logger.info(f"Recherche de chunks pertinents pour la requête : '{query}'")

    # 1. Charger le modèle d'embedding
    embedding_model = get_embedding_model()

    # 2. Embedder la requête
    query_vector = embedding_model.embed_query(query)
    logger.debug(f"Requête embeddée (taille: {len(query_vector)})")

    # 3. Se connecter à Qdrant
    client = get_qdrant_client()

    # 4. Préparer le filtre (optionnel)
    query_filter_conditions = []
    
    if video_id:
        logger.info(f"Application du filtre pour la vidéo : {video_id}")
        query_filter_conditions.append(
            FieldCondition(
                key="video_id",
                match=MatchValue(value=video_id)
            )
        )
    
    if language_code:
        logger.info(f"Application du filtre pour la langue : {language_code}")
        query_filter_conditions.append(
            FieldCondition(
                key="language",
                match=MatchValue(value=language_code)
            )
        )

    query_filter = None
    if query_filter_conditions:
        query_filter = Filter(must=query_filter_conditions)

    # 5. Rechercher dans Qdrant
    logger.info(f"Lancement de la recherche dans la collection '{collection_name}'")
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,  # Récupérer les métadonnées
        with_vectors=False   # Ne pas récupérer les vecteurs
    )

    # 6. Formater les résultats
    results = []
    for point in search_result:
        results.append({
            "score": point.score,
            "text": point.payload.get("text", ""),
            "video_id": point.payload.get("video_id", ""),
            "language": point.payload.get("language", ""),
            "chunk_index": point.payload.get("chunk_index", -1)
        })

    logger.info(f"Retrieved {len(results)} relevant chunks.")
    return results