# main.py

from src.loggings import configure_logging

logger = configure_logging(log_file="main.log", logger_name="__main__")

# --- Modules pour l'ingestion (indexation) ---
from src.youtube import extract_video_id, save_txt
from src.embedding import process_and_store_transcript_txt
from youtube_transcript_api import YouTubeTranscriptApi
import os

# --- Modules pour le RAG (recherche & génération) ---
from src.retrieve import retrieve_relevant_chunks
from src.query import answer_question_with_grok
from src.qdrant import create_video_id_index, get_qdrant_client # Ajout de get_qdrant_client

# --- Initialisation ---
ytt = YouTubeTranscriptApi()
lang = "en"
COLLECTION_NAME = "youtube_transcripts"

def ingest_video(video_url: str):
    """
    Fonction pour indexer une vidéo : récupère la transcription, la sauvegarde,
    la découpe, l'embedde et la stocke dans Qdrant.
    """
    logger.info("=== Début de l'ingestion de la vidéo ===")
    
    # 1. Extraire l'ID de la vidéo
    video_id = extract_video_id(video_url)
    if not video_id:
        logger.error("Impossible d'extraire l'ID de la vidéo.")
        return False

    logger.info(f"Traitement de la vidéo avec ID: {video_id}")

    try:
        # 2. Récupérer la transcription
        transcript = ytt.fetch(video_id=video_id, languages=[lang])
        logger.info(f"Transcription récupérée pour la vidéo {video_id}")
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la transcription pour {video_id}: {e}")
        return False

    # 3. Définir le chemin de sauvegarde du fichier TXT
    os.makedirs("./downloads", exist_ok=True)
    txt_file_name = f"{video_id}.txt"
    txt_file_path = os.path.join("./downloads", txt_file_name)

    try:
        # 4. Sauvegarder la transcription en fichier TXT
        save_txt(transcript, out_path=txt_file_name)
        logger.info(f"Transcription sauvegardée dans {txt_file_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier TXT pour {video_id}: {e}")
        return False

    try:
        # 5. Traiter le fichier TXT : découper, embedder et stocker dans Qdrant
        process_and_store_transcript_txt(
            txt_file_path=txt_file_path,
            collection_name=COLLECTION_NAME,
            video_id=video_id,
            chunk_size=700,
            chunk_overlap=100
        )
        logger.info(f"Transcription de {video_id} traitée et stockée dans Qdrant.")
        logger.info("=== Fin de l'ingestion de la vidéo ===")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du traitement et du stockage pour {video_id}: {e}")
        return False

def run_rag_pipeline(question: str, target_video_id: str = None, grok_model: str = "openai/gpt-oss-120b"):
    """
    Fonction pour exécuter le pipeline RAG complet : Retrieve -> Generate.
    
    Args:
        question (str): La question posée par l'utilisateur.
        target_video_id (str, optional): L'ID de la vidéo cible. Si None, cherche dans toutes.
        grok_model (str): Le modèle Groq à utiliser pour la génération.
    """
    logger.info("=== Début du pipeline RAG (Retrieve & Generate) ===")
    logger.info(f"Question: '{question}'")
    if target_video_id:
        logger.info(f"Filtre vidéo appliqué: {target_video_id}")

    # 1. Retrieve: Récupérer les chunks pertinents
    logger.info("Étape 1: Récupération des chunks...")
    retrieved_chunks = retrieve_relevant_chunks(
        query=question,
        collection_name=COLLECTION_NAME,
        video_id=target_video_id,
        top_k=5
    )

    if not retrieved_chunks:
        logger.warning("Aucun chunk pertinent trouvé.")
        print("\nAucun chunk pertinent trouvé pour répondre à la question.")
        logger.info("=== Fin du pipeline RAG ===")
        return

    logger.info(f"Trouvé {len(retrieved_chunks)} chunks pertinents.")

    # 2. Generate: Générer la réponse avec Grok
    logger.info("Étape 2: Génération de la réponse avec Grok...")
    answer = answer_question_with_grok(
        question=question,
        chunks=retrieved_chunks,
        model=grok_model,
        max_tokens=1000,
        temperature=0.3
    )

    # 3. Afficher les résultats
    print("\n" + "="*60)
    print(f"Question: {question}")
    print("-" * 60)
    if target_video_id:
        print(f"Filtré sur la vidéo ID: {target_video_id}")
    print("-" * 60)
    print("Chunks pertinents récupérés:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  Chunk {i+1} (Score: {chunk['score']:.4f}): {chunk['text'][:100]}...")
        print(f"    (Vidéo ID: {chunk['video_id']}, Index: {chunk['chunk_index']})")
    print("-" * 60)
    print("Réponse générée par Grok:")
    print(answer)
    print("="*60)
    logger.info("=== Fin du pipeline RAG ===")

def main():
    """
    Fonction principale.
    """
    logger.info("Hello from naive-rag!")

    # --- Configuration ---
    # Liste des URLs à ingérer. Ajoutez-en d'autres si nécessaire.
    video_urls_to_ingest = [
        "https://www.youtube.com/watch?v=94w6hPk7nkM"
        # "https://www.youtube.com/watch?v=autre_video"
    ]
    
    # Créer l'index sur video_id si nécessaire
    # A exécuter une seule fois ou si la collection est nouvelle
    qdrant_client = get_qdrant_client()
    create_video_id_index(qdrant_client, COLLECTION_NAME)

    # --- Ingestion des vidéos ---
    all_ingestions_successful = True
    for url in video_urls_to_ingest:
        success = ingest_video(url)
        if not success:
            all_ingestions_successful = False
            logger.error(f"L'ingestion de la vidéo {url} a échoué.")
            
    if not all_ingestions_successful:
        logger.error("Certaines ingestions ont échoué. Arrêt.")
        return

    # --- Exécution du pipeline RAG ---
    # Vous pouvez modifier ces valeurs pour tester différentes questions/vidéos
    test_question = "Explique le concept principal discuté dans la vidéo."
    # Utiliser l'ID de la première vidéo pour le filtrage, ou None pour toutes
    test_video_id = extract_video_id(video_urls_to_ingest[0]) 
    # Assurez-vous que ce modèle est disponible sur Groq
    test_model = "openai/gpt-oss-120b" 

    run_rag_pipeline(
        question=test_question,
        target_video_id=test_video_id,
        grok_model=test_model
    )

if __name__ == "__main__":
    main()