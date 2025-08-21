# src/mongo_utils.py

import os
import uuid
import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from .loggings import configure_logging
from dotenv import load_dotenv

load_dotenv()

logger = configure_logging(log_file="mongo.log", logger_name="__mongo__")

class ConversationManager:
    """
    Gère l'enregistrement et la récupération des conversations dans MongoDB.
    """
    
    def __init__(self):
        """Initialise la connexion à MongoDB."""
        self.mongo_uri = os.getenv("MONGO_DB_URI_RAG")
        self.db_name = os.getenv("MONGO_DB_NAME_RAG")
        self.collection_name = "conversations"
        
        try:
            if not self.mongo_uri or not self.db_name:
                raise ValueError("MONGO_DB_URI_RAG et MONGO_DB_NAME_RAG doivent être définis")
            logger.info(f"Connexion à MongoDB: {self.mongo_uri}")
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000
            )
            self.client.admin.command('ping')
            logger.info("Connexion MongoDB établie avec succès")
            
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            # Créer les index nécessaires
            self.collection.create_index("session_id", unique=True)
            self.collection.create_index("video_id")
            self.collection.create_index("created_at")
            logger.debug(f"Index créés sur la collection {self.collection_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Échec de la connexion à MongoDB: {e}")
            raise ConnectionError("Impossible de se connecter à la base de données MongoDB") from e
        except PyMongoError as e:
            logger.error(f"Erreur MongoDB lors de l'initialisation: {e}")
            raise
    
    def generate_session_id(self) -> str:
        """Génère un ID de session unique."""
        return str(uuid.uuid4())
    
    def create_conversation(self, video_id: str, messages: list = None, metadata: dict = None, session_id: str = None) -> str:
        """
        Crée une nouvelle conversation dans MongoDB.
        
        Args:
            video_id (str): ID de la vidéo YouTube associée
            messages (list, optional): Liste initiale des messages
            metadata (dict, optional): Métadonnées supplémentaires
            session_id (str, optional): ID de session à utiliser. Si None, en génère un nouveau.
            
        Returns:
            str: L'ID de session utilisé
        """
        # Utiliser l'ID fourni ou en générer un nouveau
        if session_id is None:
            session_id = self.generate_session_id()
        
        timestamp = datetime.datetime.utcnow()
        
        if messages is None:
            messages = []
        
        # Préparer les métadonnées
        conversation_metadata = {
            "session_id": session_id, 
            "video_id": video_id,
            "start_time": timestamp,
            "model_used": os.getenv("DEFAULT_MODEL", "openai/gpt-oss-120b"),
            "embedding_model": os.getenv("DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
            "response_language": os.getenv("DEFAULT_RESPONSE_LANGUAGE", "English")
        }
        
        if metadata:
            conversation_metadata.update(metadata)
        
        # Nettoyer les messages pour MongoDB
        cleaned_messages = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                cleaned_msg = {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": datetime.datetime.utcnow()
                }
                cleaned_messages.append(cleaned_msg)
        
        # Créer le document de conversation
        conversation = {
            "session_id": session_id,
            "video_id": video_id,
            "messages": cleaned_messages,
            "metadata": conversation_metadata,
            "created_at": timestamp,
            "last_updated": timestamp
        }
        
        try:
            logger.info(f"Création de la conversation {session_id} pour la vidéo {video_id}")
            result = self.collection.insert_one(conversation)
            if result.inserted_id:
                logger.info(f"Conversation {session_id} créée avec succès")
                return session_id
            else:
                logger.error(f"Échec de la création de la conversation {session_id}")
                raise Exception("Failed to create conversation")
        except PyMongoError as e:
            logger.error(f"Erreur lors de la création de la conversation: {e}")
            raise
    
    def add_messages_to_conversation(self, session_id: str, new_messages: list):
        """
        Ajoute des messages à une conversation existante.
        
        Args:
            session_id (str): ID de la session
            new_messages (list): Messages à ajouter
        """
        if not new_messages:
            logger.debug("Aucun message à ajouter")
            return
            
        # Préparer les nouveaux messages
        cleaned_messages = []
        for msg in new_messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                cleaned_msg = {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": datetime.datetime.utcnow()
                }
                cleaned_messages.append(cleaned_msg)
        
        try:
            logger.info(f"Ajout de {len(cleaned_messages)} messages à la conversation {session_id}")
            result = self.collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"messages": {"$each": cleaned_messages}},
                    "$set": {"last_updated": datetime.datetime.utcnow()}
                }
            )
            
            if result.matched_count > 0:
                logger.info(f"Messages ajoutés à la conversation {session_id}")
            else:
                logger.warning(f"Conversation {session_id} non trouvée pour l'ajout de messages")
                # Optionnel : créer la conversation si elle n'existe pas
                # self.create_conversation(session_id, new_messages)
                
        except PyMongoError as e:
            logger.error(f"Erreur lors de l'ajout de messages à la conversation: {e}")
            raise
    
    def get_conversation(self, session_id: str) -> dict:
        """Récupère une conversation spécifique."""
        try:
            logger.info(f"Récupération de la conversation {session_id}")
            conversation = self.collection.find_one({"session_id": session_id})
            if conversation:
                logger.info(f"Conversation {session_id} récupérée avec succès")
                return conversation
            else:
                logger.warning(f"Conversation {session_id} non trouvée")
                return None
        except PyMongoError as e:
            logger.error(f"Erreur lors de la récupération de la conversation: {e}")
            raise
    
    def get_video_conversations(self, video_id: str, limit: int = 10) -> list:
        """Récupère les conversations associées à une vidéo."""
        try:
            logger.info(f"Récupération des conversations pour la vidéo {video_id} (limite: {limit})")
            conversations = list(self.collection.find(
                {"video_id": video_id}
            ).sort("created_at", -1).limit(limit))
            
            logger.info(f"{len(conversations)} conversations récupérées pour la vidéo {video_id}")
            return conversations
        except PyMongoError as e:
            logger.error(f"Erreur lors de la récupération des conversations de la vidéo: {e}")
            raise
    
    def delete_conversation(self, session_id: str) -> bool:
        """Supprime une conversation spécifique."""
        try:
            logger.info(f"Suppression de la conversation {session_id}")
            result = self.collection.delete_one({"session_id": session_id})
            if result.deleted_count > 0:
                logger.info(f"Conversation {session_id} supprimée avec succès")
                return True
            else:
                logger.warning(f"Conversation {session_id} non trouvée pour suppression")
                return False
        except PyMongoError as e:
            logger.error(f"Erreur lors de la suppression de la conversation: {e}")
            raise
    
    def close_connection(self):
        """Ferme la connexion MongoDB."""
        logger.info("Fermeture de la connexion MongoDB")
        if self.client:
            self.client.close()