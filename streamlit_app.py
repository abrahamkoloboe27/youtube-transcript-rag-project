# streamlit_app.py - Version complète mise à jour

import streamlit as st
from src.youtube import extract_video_id
from src.embedding import process_and_store_transcript_txt
from src.qdrant import get_qdrant_client, check_video_exists
from src.retrieve import retrieve_relevant_chunks
from src.query import answer_question_with_grok
from src.embedding import get_embedding_model
from youtube_transcript_api import YouTubeTranscriptApi
from src.loggings import configure_logging
import os
import time

# Configuration du logging
logger = configure_logging(log_file="streamlit_app.log", logger_name="__streamlit_app__")
logger.info("Démarrage de l'application Streamlit")

# Configuration de la page
st.set_page_config(
    page_title="Naive RAG YouTube",
    page_icon="🎥",
    layout="wide"
)

# === CACHING DES MODELES ===
@st.cache_resource
def load_embedding_model():
    """Charge le modèle d'embedding au démarrage et le garde en cache"""
    with st.spinner("Chargement du modèle d'embedding..."):
        logger.info("Chargement du modèle d'embedding...")
        model = get_embedding_model()
        logger.info("Modèle d'embedding chargé avec succès")
        return model

@st.cache_resource
def get_qdrant_client_cached():
    """Cache le client Qdrant"""
    logger.info("Initialisation du client Qdrant...")
    client = get_qdrant_client()
    logger.info("Client Qdrant initialisé")
    return client

# === CONSTANTES ===
COLLECTION_NAME = "youtube_transcripts"
AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b", 
    "qwen/qwen3-32b"
]
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
DEFAULT_LANGUAGE = "en"

# === INITIALISATION ===
if 'messages' not in st.session_state:
    st.session_state.messages = []
    logger.debug("Initialisation de st.session_state.messages")
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None
    logger.debug("Initialisation de st.session_state.current_video_id")
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
    logger.debug("Initialisation de st.session_state.video_processed")
if 'temperature' not in st.session_state:
    st.session_state.temperature = DEFAULT_TEMPERATURE
    logger.debug("Initialisation de st.session_state.temperature")
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    logger.debug("Initialisation de st.session_state.max_tokens")
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = DEFAULT_LANGUAGE
    logger.debug("Initialisation de st.session_state.selected_language")
if 'available_languages' not in st.session_state:
    st.session_state.available_languages = []
    logger.debug("Initialisation de st.session_state.available_languages")

# Charger les modèles au démarrage
try:
    embedding_model = load_embedding_model()
    qdrant_client = get_qdrant_client_cached()
    ytt = YouTubeTranscriptApi()
    logger.info("Tous les modèles et clients chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement des modèles/clients: {e}")
    st.error("Erreur lors du chargement de l'application. Veuillez consulter les logs.")

# === SIDEBAR ===
with st.sidebar:
    st.title("🎥 Naive RAG YouTube")
    st.markdown("---")
    
    # Input URL
    youtube_url = st.text_input("🔗 URL YouTube", placeholder="https://www.youtube.com/watch?v=...")

    # Ajouter un sélecteur de langue pour la réponse (pas pour l'ingestion)
    response_language = st.selectbox(
        "🌍 Langue de réponse",
        options=["Français", "English", "Español", "Deutsch"],
        index=0
    )

    # Convertir en code langue
    language_codes = {"Français": "fr", "English": "en", "Español": "es", "Deutsch": "de"}
    selected_response_language = language_codes[response_language]

    # Sélecteur de modèle
    selected_model = st.selectbox(
        "🤖 Modèle de réponse",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0
    )
 

    
    st.markdown("---")
    
    # Paramètres de génération
    st.subheader("⚙️ Paramètres de génération")
    st.session_state.temperature = st.slider(
        "Température", 
        min_value=0.0, 
        max_value=1.0, 
        value=float(st.session_state.temperature), 
        step=0.1,
        help="Contrôle la créativité de la réponse (0 = déterministe, 1 = créatif)"
    )
    
    st.session_state.max_tokens = st.slider(
        "Max Tokens", 
        min_value=100, 
        max_value=2000, 
        value=int(st.session_state.max_tokens), 
        step=100,
        help="Longueur maximale de la réponse"
    )
    
    st.markdown("---")
    
    # Dans streamlit_app.py - Remplacer la section de détection des langues par :

    if youtube_url:
        logger.info(f"Traitement de l'URL YouTube: {youtube_url}")
        
        # Extraire l'ID de la vidéo
        video_id = extract_video_id(youtube_url)
        
        if not video_id:
            st.error("❌ URL YouTube invalide")
            logger.warning(f"URL YouTube invalide fournie: {youtube_url}")
        else:
            # Vérifier si la vidéo existe déjà (dans la langue par défaut ou la langue sélectionnée)
            logger.info(f"Vérification de l'existence de la vidéo {video_id} dans Qdrant...")
            video_exists = check_video_exists(qdrant_client, COLLECTION_NAME, video_id)
            
            if video_exists:
                st.success(f"✅ Vidéo déjà traitée (ID: {video_id})")
                logger.info(f"Vidéo {video_id} déjà présente dans la base")
                st.session_state.current_video_id = video_id
                st.session_state.video_processed = True
            else:
                st.warning("🔄 Vidéo non traitée - Lancement de l'ingestion...")
                logger.info(f"Vidéo {video_id} non trouvée, démarrage de l'ingestion...")
                
                # Processus d'ingestion - garder l'approche originale
                try:
                    with st.spinner("📥 Récupération de la transcription..."):
                        logger.info(f"Récupération de la transcription pour {video_id}")
                        # Utiliser l'approche originale
                        transcript = ytt.fetch(video_id=video_id, languages=['en', 'fr'])  # ou juste ['en']
                        logger.info(f"Transcription récupérée ({len(transcript)} segments)")
                    
                    with st.spinner("💾 Sauvegarde de la transcription..."):
                        logger.info(f"Sauvegarde de la transcription pour {video_id}")
                        os.makedirs("./downloads", exist_ok=True)
                        txt_file_name = f"{video_id}.txt"
                        txt_file_path = f"./downloads/{txt_file_name}"
                        
                        try:
                            from src.youtube import save_txt
                            save_txt(transcript, out_path=txt_file_name)
                            logger.info(f"Transcription sauvegardée dans {txt_file_path}")
                        except Exception as e:
                            logger.error(f"Erreur lors de la sauvegarde de la transcription pour {video_id}: {e}")
                            raise
                    
                    with st.spinner("🧠 Traitement et stockage dans Qdrant..."):
                        logger.info(f"Traitement et stockage de {video_id} dans Qdrant")
                        # Utiliser la version originale sans language_code
                        from src.embedding import process_and_store_transcript_txt
                        process_and_store_transcript_txt(
                            txt_file_path=txt_file_path,
                            collection_name=COLLECTION_NAME,
                            video_id=video_id,
                            chunk_size=700,
                            chunk_overlap=100
                        )
                        
                    with st.spinner("🔍 Vérification du stockage..."):
                        video_stored = check_video_exists(qdrant_client, COLLECTION_NAME, video_id)
                        if video_stored:
                            logger.info(f"Vidéo {video_id} confirmée dans Qdrant")
                            st.success("✅ Vidéo traitée et stockée avec succès!")
                            st.session_state.current_video_id = video_id
                            st.session_state.video_processed = True
                        else:
                            logger.warning(f"Vidéo {video_id} non trouvée dans Qdrant après ingestion")
                            st.warning("⚠️ Problème lors du stockage - aucun chunk trouvé")
                            st.session_state.video_processed = False
                    
                except Exception as e:
                    error_msg = f"❌ Erreur lors du traitement: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Erreur lors du traitement de la vidéo {video_id}: {e}")
                    st.session_state.video_processed = False
# === CHAT INTERFACE ===
st.title("💬 Chat avec votre vidéo YouTube")

# Afficher l'historique des messages
logger.debug(f"Affichage de {len(st.session_state.messages)} messages dans l'historique")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utilisateur
if prompt := st.chat_input("Posez votre question sur la vidéo...", 
                          disabled=not st.session_state.video_processed):
    
    logger.info(f"Question utilisateur: {prompt}")
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Réponse de l'assistant
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            if not st.session_state.current_video_id:
                full_response = "❌ Aucune vidéo sélectionnée. Veuillez entrer une URL YouTube dans la sidebar."
                logger.warning("Tentative de question sans vidéo sélectionnée")
            else:
                # Récupérer les chunks pertinents avec la langue sélectionnée
                with st.spinner("🔍 Recherche des informations pertinentes..."):
                    logger.info(f"Recherche de chunks pertinents pour: {prompt}")
                    retrieved_chunks = retrieve_relevant_chunks(
                        query=prompt,
                        collection_name=COLLECTION_NAME,
                        video_id=st.session_state.current_video_id,
                        top_k=10
                    )
                    logger.info(f"Trouvé {len(retrieved_chunks)} chunks pertinents")
                
                if not retrieved_chunks:
                    full_response = "❌ Je n'ai trouvé aucune information pertinente dans la vidéo pour répondre à votre question."
                    logger.info("Aucun chunk pertinent trouvé pour la requête")
                else:
                    with st.spinner("🤖 Génération de la réponse..."):
                        logger.info(f"Génération de réponse avec le modèle {selected_model}")
                        # Vous pouvez ajouter une instruction dans le prompt pour la langue de réponse
                        prompt_with_language = f"Réponds en {response_language}: {prompt}"
                        
                        full_response = answer_question_with_grok(
                            question=prompt_with_language,
                            chunks=retrieved_chunks,
                            model=selected_model,
                            max_tokens=st.session_state.max_tokens,
                            temperature=st.session_state.temperature,
                            conversation_history=st.session_state.messages
                        )
            
            # Afficher la réponse progressivement (effet de frappe)
            for chunk in full_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_response = f"❌ Une erreur s'est produite: {str(e)}"
            message_placeholder.markdown(error_response)
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            full_response = error_response
    
    # Ajouter la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    logger.debug("Réponse ajoutée à l'historique")

# Bouton pour réinitialiser la conversation
if st.sidebar.button("🗑️ Réinitialiser la conversation"):
    logger.info("Réinitialisation de la conversation demandée")
    st.session_state.messages = []
    st.session_state.current_video_id = None
    st.session_state.video_processed = False
    st.session_state.selected_language = DEFAULT_LANGUAGE
    st.session_state.available_languages = []
    st.rerun()

logger.info("Fin du rendu de l'application Streamlit")