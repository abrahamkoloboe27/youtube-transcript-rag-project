# streamlit_app.py

import streamlit as st
from src.youtube import extract_video_id, save_txt
from src.embedding import process_and_store_transcript_txt, get_embedding_model
from src.mongo_utils import ConversationManager
from src.qdrant import get_qdrant_client, check_video_exists
from src.retrieve import retrieve_relevant_chunks 
from src.query import answer_question_with_grok
from youtube_transcript_api import YouTubeTranscriptApi
from src.loggings import configure_logging
from src.loggings import set_client_info_function
import os
import time


# === RÉCUPÉRATION DES INFORMATIONS CLIENT ===
def get_streamlit_client_info():
    """
    Récupère les informations du client depuis Streamlit.
    Note: Streamlit limite l'accès à certaines informations pour des raisons de sécurité.
    """
    try:
        # Tentative de récupération de l'IP (limitée sur Streamlit Cloud)
        client_ip = "unknown"  # Streamlit Cloud masque souvent l'IP réelle
        
        # Récupération de l'user agent via st.context (si disponible)
        user_agent = "unknown"
        try:
            # Méthode pour les versions récentes de Streamlit
            from streamlit.runtime import get_instance
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            
            ctx = get_script_run_ctx()
            if ctx and hasattr(ctx, 'session_id'):
                # L'IP n'est généralement pas accessible directement
                # On peut utiliser l'ID de session comme identifiant
                session_info = get_instance()._session_info_by_id.get(ctx.session_id)
                if session_info:
                    # Certaines informations peuvent être disponibles dans session_info
                    pass
        except:
            pass
            
        return {
            'client_ip': client_ip,
            'user_agent': st.context.headers.get('User-Agent', 'unknown') if hasattr(st, 'context') and hasattr(st.context, 'headers') else 'unknown',
            'session_id': st.session_state.get('session_id', 'unknown'),
            'video_id': st.session_state.get('current_video_id', 'unknown')
        }
    except Exception as e:
        return {
            'client_ip': 'unknown',
            'user_agent': 'unknown',
            'session_id': st.session_state.get('session_id', 'unknown'),
            'video_id': st.session_state.get('current_video_id', 'unknown'),
            'error': str(e)
        }


# === CONFIGURATION DU LOGGING CLIENT INFO ===
# Définir la fonction qui récupère les infos client
set_client_info_function(get_streamlit_client_info)


# Configuration du logging
logger = configure_logging(log_file="streamlit_app.log", logger_name="__streamlit_app__")
logger.info("Démarrage de l'application Streamlit")

# Configuration de la page
st.set_page_config(
    page_title="Naive RAG YouTube",
    page_icon="🎥",
    layout="wide"
)

# === INITIALISATION DE LA SESSION MONGO DB ===
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'conversation_manager' not in st.session_state:
    try:
        st.session_state.conversation_manager = ConversationManager()
        logger.info("ConversationManager initialisé")
    except Exception as e:
        logger.error(f"Impossible d'initialiser ConversationManager: {e}")
        st.session_state.conversation_manager = None

# === CACHING DES MODELES ===
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
    "qwen/qwen3-32b",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
]
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Modèles d'embedding disponibles
EMBEDDING_MODELS = {
    "sentence-transformers/all-mpnet-base-v2": "Default (MPNet)",
    "Qwen3-Embedding-0.6B": "Qwen3 0.6B",
    "bilingual-embedding-large": "Bilingual Large"
}
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
DEFAULT_RESPONSE_LANGUAGE = "English"

# === INITIALISATION DES VARIABLES DE SESSION ===
session_keys_defaults = {
    'messages': [],
    'current_video_id': None,
    'video_processed': False,
    'temperature': DEFAULT_TEMPERATURE,
    'max_tokens': DEFAULT_MAX_TOKENS,
    'selected_embedding_model': DEFAULT_EMBEDDING_MODEL,
    'response_language': DEFAULT_RESPONSE_LANGUAGE
}

for key, default_value in session_keys_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
        logger.debug(f"Initialisation de st.session_state.{key} à {default_value}")

# Charger les clients au démarrage
try:
    qdrant_client = get_qdrant_client_cached()
    ytt = YouTubeTranscriptApi()
    logger.info("Clients Qdrant et YouTubeTranscriptApi initialisés avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation des clients: {e}")
    st.error("Error initializing application. Please check the logs.")

# === SIDEBAR ===
with st.sidebar:
    st.title("🎥 Naive RAG YouTube")
    st.markdown("---")
    
    # Input URL
    youtube_url = st.text_input("🔗 YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    # Sélecteur de langue pour la réponse
    st.session_state.response_language = st.selectbox(
        "🌍 Response Language",
        options=["English", "Français", "Español", "Deutsch", "Hindi", "Telugu"],
        index=0,
        help="Select the language for the response"
    )

    # Sélecteur de modèle de réponse (LLM)
    selected_model = st.selectbox(
        "🤖 Response Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0
    )
    
    st.markdown("---")
    
    # Paramètres de génération
    st.subheader("⚙️ Generation Settings")
    st.session_state.temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=float(st.session_state.temperature), 
        step=0.1,
        help="Controls randomness (0 = deterministic, 1 = creative)"
    )
    
    st.session_state.max_tokens = st.slider(
        "Max Tokens", 
        min_value=100, 
        max_value=2500, 
        value=int(st.session_state.max_tokens), 
        step=100,
        help="Maximum length of the response"
    )
    
    st.markdown("---")
    
    # Traitement de l'URL YouTube
    if youtube_url:
        logger.info(f"Processing YouTube URL: {youtube_url}")
        
        # Extraire l'ID de la vidéo
        video_id = extract_video_id(youtube_url)
        
        if not video_id:
            st.error("❌ Invalid YouTube URL")
            logger.warning(f"Invalid YouTube URL provided: {youtube_url}")
        else:
            # Vérifier si la vidéo existe déjà
            logger.info(f"Checking existence of video {video_id} in Qdrant...")
            video_exists = check_video_exists(qdrant_client, COLLECTION_NAME, video_id)
            
            if video_exists:
                st.success(f"✅ Video already processed (ID: {video_id})")
                logger.info(f"Video {video_id} already exists in the database")
                st.session_state.current_video_id = video_id
                st.session_state.video_processed = True
            else:
                st.info("🔄 Video not processed - Starting ingestion...")
                logger.info(f"Video {video_id} not found, starting ingestion...")
                
                # Processus d'ingestion
                try:
                    with st.spinner("📥 Fetching transcript..."):
                        logger.info(f"Fetching transcript for {video_id}")
                        try:
                            # Essayer plusieurs langues courantes
                            transcript_languages = ['en', 'fr', 'es', 'de', 'hi', 'te']
                            transcript = None
                            for lang in transcript_languages:
                                try:
                                    transcript = ytt.fetch(video_id=video_id, languages=[lang])
                                    logger.info(f"Transcript fetched in language '{lang}'")
                                    break
                                except:
                                    continue
                            
                            if not transcript:
                                # Essayer sans spécifier de langue (auto-détection)
                                transcript = ytt.fetch(video_id=video_id)
                                logger.info(f"Transcript fetched with auto-detected language")
                            
                            logger.info(f"Transcript retrieved ({len(transcript)} segments)")
                        except Exception as e:
                            error_msg = str(e)
                            if "YouTube is blocking requests from your IP" in error_msg or "IP has been blocked" in error_msg:
                                st.error("""
                                    ⚠️ **YouTube is blocking access from cloud servers**
                                    
                                    This is a known limitation when using YouTube transcripts from cloud platforms like Streamlit.
                                    
                                    **Possible solutions:**
                                    1. Try with a different video that has manually added subtitles
                                    2. Use a local instance of the app (run Streamlit on your own computer)
                                    3. Consider alternative sources for your RAG system
                                    
                                    *Note: This is not an issue with the app itself but with YouTube's restrictions on cloud servers.*
                                """)
                                logger.error(f"YouTube IP blocking detected for video {video_id}: {error_msg}")
                                st.stop()
                            else:
                                raise
                    
                    with st.spinner("💾 Saving transcript..."):
                        logger.info(f"Saving transcript for {video_id}")
                        os.makedirs("./downloads", exist_ok=True)
                        txt_file_name = f"{video_id}.txt"
                        txt_file_path = f"./downloads/{txt_file_name}"
                        
                        try:
                            save_txt(transcript, out_path=txt_file_name)
                            logger.info(f"Transcript saved to {txt_file_path}")
                        except Exception as e:
                            logger.error(f"Error saving transcript for {video_id}: {e}")
                            raise
                    
                    with st.spinner("🧠 Processing and storing in Qdrant..."):
                        logger.info(f"Processing and storing {video_id} in Qdrant using model {st.session_state.selected_embedding_model}")
                        process_and_store_transcript_txt(
                            txt_file_path=txt_file_path,
                            collection_name=COLLECTION_NAME,
                            video_id=video_id,
                            embedding_model_name=st.session_state.selected_embedding_model,
                            chunk_size=700,
                            chunk_overlap=100
                        )
                        
                    with st.spinner("🔍 Verifying storage..."):
                        video_stored = check_video_exists(qdrant_client, COLLECTION_NAME, video_id)
                        if video_stored:
                            logger.info(f"Video {video_id} confirmed in Qdrant")
                            st.success("✅ Video processed and stored successfully!")
                            st.session_state.current_video_id = video_id
                            st.session_state.video_processed = True
                        else:
                            logger.warning(f"Video {video_id} not found in Qdrant after ingestion")
                            st.warning("⚠️ Storage issue - no chunks found")
                            st.session_state.video_processed = False
                
                except Exception as e:
                    error_msg = f"❌ Error during processing: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error processing video {video_id}: {e}")
                    st.session_state.video_processed = False

# === CHAT INTERFACE ===
st.title("💬 Chat with your YouTube Video")

# Afficher l'historique des messages
logger.debug(f"Displaying {len(st.session_state.messages)} messages from history")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utilisateur
if prompt := st.chat_input("Ask a question about the video...", 
                          disabled=not st.session_state.video_processed):
    
    logger.info(f"Received user prompt: {prompt}")
    
    # Générer un ID de session ET créer la conversation si ce n'est pas déjà fait
    if not st.session_state.session_id:
        if st.session_state.conversation_manager:
            st.session_state.session_id = st.session_state.conversation_manager.generate_session_id()
            logger.info(f"Nouvelle session créée: {st.session_state.session_id}")
            
            # Tenter de créer la conversation initiale
            try:
                session_id_created = st.session_state.conversation_manager.create_conversation(
                    video_id=st.session_state.current_video_id or "unknown",
                    messages=[], # Commencer avec une liste vide
                    metadata={
                        "response_language": st.session_state.response_language,
                        "model_used": selected_model,
                        "temperature": st.session_state.temperature,
                        "max_tokens": st.session_state.max_tokens
                    },
                    session_id=st.session_state.session_id  # Passer l'ID existant
                )
                # Vérifier si l'ID retourné correspond (bonne pratique défensive)
                if session_id_created == st.session_state.session_id:
                    logger.info(f"✅ Conversation initiale créée avec succès pour la session {st.session_state.session_id}")
                else:
                    logger.warning(f"⚠️ Incohérence d'ID de session: généré={st.session_state.session_id}, créé={session_id_created}")
            except Exception as e:
                error_msg = f"❌ Échec de la création de la conversation initiale: {e}"
                logger.error(error_msg)
                # Optionnel: Afficher l'erreur à l'utilisateur
                # st.sidebar.error(error_msg)
        else:
            logger.warning("ConversationManager non disponible lors de la création de session")
    
    logger.info(f"User question: {prompt}")
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
                full_response = "❌ No video selected. Please enter a YouTube URL in the sidebar."
                logger.warning("Question attempt without selected video")
            else:
                # Récupérer les chunks pertinents
                with st.spinner("🔍 Searching for relevant information..."):
                    logger.info(f"Searching for relevant chunks for: {prompt}")
                    retrieved_chunks = retrieve_relevant_chunks(
                        query=prompt,
                        collection_name=COLLECTION_NAME,
                        video_id=st.session_state.current_video_id,
                        embedding_model_name=st.session_state.selected_embedding_model,
                        top_k=10
                    )
                    logger.info(f"Found {len(retrieved_chunks)} relevant chunks")
                
                if not retrieved_chunks:
                    full_response = "❌ I couldn't find any relevant information in the video to answer your question."
                    logger.info("No relevant chunks found for the query")
                else:
                    # Générer la réponse avec l'historique de conversation
                    with st.spinner("🤖 Generating response..."):
                        logger.info(f"Generating response with model {selected_model}")
                        
                        # Ajouter une instruction sur la langue de réponse dans le prompt
                        language_instruction = ""
                        if st.session_state.response_language and st.session_state.response_language != "English":
                            language_instruction = f"Please answer in {st.session_state.response_language}. "
                        
                        contextualized_prompt = f"{language_instruction}{prompt}"
                        
                        full_response = answer_question_with_grok(
                            question=contextualized_prompt,
                            chunks=retrieved_chunks,
                            model=selected_model,
                            max_tokens=st.session_state.max_tokens,
                            temperature=st.session_state.temperature,
                            conversation_history=st.session_state.messages
                        )
                        logger.info("Response generated successfully")
            
            # Afficher la réponse progressivement (effet de frappe)
            for chunk in full_response.split():
                full_response += chunk + " "
                time.sleep(0.02) 
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_response = f"❌ An error occurred: {str(e)}"
            message_placeholder.markdown(error_response)
            logger.error(f"Error generating response: {e}")
            full_response = error_response
    
    # Ajouter la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    logger.debug("Response added to history")
    
    # Sauvegarder la conversation dans MongoDB
    if st.session_state.conversation_manager and st.session_state.session_id and full_response:
        try:
            # Ajouter les deux nouveaux messages (user + assistant)
            new_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": full_response}
            ]
            st.session_state.conversation_manager.add_messages_to_conversation(
                st.session_state.session_id, 
                new_messages
            )
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la conversation: {e}")

# Bouton pour réinitialiser la conversation
if st.sidebar.button("🗑️ Reset Conversation"):
    logger.info("Conversation reset requested")
    
    # Réinitialiser les messages et générer un nouvel ID de session
    st.session_state.messages = []
    
    if st.session_state.conversation_manager:
        # Générer un nouvel ID de session
        old_session_id = st.session_state.session_id
        st.session_state.session_id = st.session_state.conversation_manager.generate_session_id()
        logger.info(f"Nouvelle session créée après réinitialisation: {st.session_state.session_id}")
        
        # Créer la nouvelle conversation vide
        try:
            st.session_state.conversation_manager.create_conversation(
                video_id=st.session_state.current_video_id or "unknown",
                messages=[],
                metadata={
                    "response_language": st.session_state.response_language,
                    "model_used": selected_model,
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                session_id=st.session_state.session_id  # Passer l'ID existant
            )
            logger.info(f"Nouvelle conversation créée pour la session {st.session_state.session_id}")
        except Exception as e:
            logger.error(f"Erreur lors de la création de la nouvelle conversation: {e}")
    
    st.rerun()

# === DEBUG ===
if st.sidebar.checkbox("🔍 Debug Conversation"):
    st.write("### État de la session")
    st.write(f"- Session ID: `{st.session_state.session_id}`")
    st.write(f"- Nombre de messages: `{len(st.session_state.messages)}`")
    
    if st.session_state.session_id and st.session_state.conversation_manager:
        try:
            conv = st.session_state.conversation_manager.get_conversation(st.session_state.session_id)
            if conv:
                st.success("✅ Conversation trouvée dans MongoDB")
                st.json(conv) # Affiche le contenu brut
            else:
                st.warning("⚠️ Session ID présent mais conversation NON trouvée dans MongoDB")
        except Exception as e:
            st.error(f"Erreur lors du debug: {e}")

logger.info("End of Streamlit application render")