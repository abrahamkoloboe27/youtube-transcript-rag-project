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
    
    Utilisation typique :
    manager = ConversationManager()
    manager.save_conversation(video_id, st.session_state.messages, metadata)
    """
    
    def __init__(self):
        """Initialise la connexion à MongoDB."""
        self.mongo_uri = os.getenv("MONGO_DB_URI_RAG")
        self.db_name = os.getenv("MONGO_DB_NAME_RAG")
        self.collection_name = "conversations"
        
        try:
            if not self.mongo_uri or not self.db_name:
                raise ValueError("MONGO_DB_URI_RAG et MONGO_DB_NAME_RAG doivent être définis")
            #logger.info(f"Connexion à MongoDB: {self.mongo_uri}")
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000  # Timeout après 5s
            )
            # Vérifier la connexion
            self.client.admin.command('ping')
            logger.info("Connexion MongoDB établie avec succès")
            
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            # Créer un index sur session_id pour les recherches rapides
            self.collection.create_index("session_id", unique=True)
            logger.debug(f"Index 'session_id' créé sur la collection {self.collection_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Échec de la connexion à MongoDB: {e}")
            raise ConnectionError("Impossible de se connecter à la base de données MongoDB") from e
        except PyMongoError as e:
            logger.error(f"Erreur MongoDB lors de l'initialisation: {e}")
            raise
    
    def generate_session_id(self) -> str:
        """Génère un ID de session unique."""
        return str(uuid.uuid4())
    
    def save_conversation(self, video_id: str, messages: list, metadata: dict = None) -> str:
        """
        Sauvegarde une conversation dans MongoDB.
        
        Args:
            video_id (str): ID de la vidéo YouTube associée
            messages (list): Liste des messages de la conversation
            metadata (dict, optional): Métadonnées supplémentaires
            
        Returns:
            str: L'ID de session généré
        """
        session_id = self.generate_session_id()
        timestamp = datetime.datetime.utcnow()
        
        # Préparer les métadonnées
        conversation_metadata = {
            "video_id": video_id,
            "start_time": timestamp,
            "model_used": os.getenv("DEFAULT_MODEL", "openai/gpt-oss-120b"),
            "embedding_model": os.getenv("DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
            "response_language": "English"  # Par défaut, sera mis à jour plus tard
        }
        
        if metadata:
            conversation_metadata.update(metadata)
        
        # Nettoyer les messages pour MongoDB
        # (Supprimer les clés non sérialisables)
        cleaned_messages = []
        for msg in messages:
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
            logger.info(f"Sauvegarde de la conversation {session_id} pour la vidéo {video_id}")
            self.collection.insert_one(conversation)
            logger.info(f"Conversation {session_id} sauvegardée avec succès")
            return session_id
        except PyMongoError as e:
            logger.error(f"Erreur lors de la sauvegarde de la conversation: {e}")
            raise
    
    def update_conversation(self, session_id: str, new_messages: list):
        """
        Met à jour une conversation existante avec de nouveaux messages.
        
        Args:
            session_id (str): ID de la session à mettre à jour
            new_messages (list): Nouveaux messages à ajouter
        """
        if not new_messages:
            return
            
        # Préparer les nouveaux messages
        cleaned_messages = []
        for msg in new_messages:
            cleaned_msg = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": datetime.datetime.utcnow()
            }
            cleaned_messages.append(cleaned_msg)
        
        try:
            logger.info(f"Mise à jour de la conversation {session_id} avec {len(new_messages)} nouveaux messages")
            logger.info(f"Nouveaux messages: {new_messages}")
            self.collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"messages": {"$each": cleaned_messages}},
                    "$set": {"last_updated": datetime.datetime.utcnow()}
                }
            )
            logger.info(f"Conversation {session_id} mise à jour avec succès")
        except PyMongoError as e:
            logger.error(f"Erreur lors de la mise à jour de la conversation: {e}")
            raise
    
    def get_conversation(self, session_id: str) -> dict:
        """
        Récupère une conversation spécifique.
        
        Args:
            session_id (str): ID de la session à récupérer
            
        Returns:
            dict: La conversation ou None si non trouvée
        """
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
        """
        Récupère les conversations associées à une vidéo spécifique.
        
        Args:
            video_id (str): ID de la vidéo
            limit (int): Nombre maximum de conversations à retourner
            
        Returns:
            list: Liste des conversations
        """
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
        """
        Supprime une conversation spécifique.
        
        Args:
            session_id (str): ID de la session à supprimer
            
        Returns:
            bool: True si supprimé, False sinon
        """
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
        self.client.close()