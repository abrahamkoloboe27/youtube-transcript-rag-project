# src/loggings.py
"""
Module de configuration centralisée du logging.
Supporte la journalisation vers console, fichier et MongoDB.
"""

import os
import logging
from log4mongo.handlers import BufferedMongoHandler

# Configuration depuis les variables d'environnement
MONGO_URI = os.environ.get("MONGO_DB_URI_RAG")
MONGO_DB = os.environ.get("MONGO_DB_NAME_RAG", "naive_rag")
MONGO_COLLECTION = "application_logs"

def configure_logging(
    log_file: str,
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    logger_name: str = __name__,
    mongo_enabled: bool = True,
    console_enabled: bool = True,
    file_enabled: bool = True,
) -> logging.Logger:
    """
    Configure un logger avec des handlers pour MongoDB, fichier et console.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Éviter les doublons si le logger est déjà configuré
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console Handler
    if console_enabled and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler
    if file_enabled and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # MongoDB Handler
    if mongo_enabled and MONGO_URI:
        try:
            mongo_handler = BufferedMongoHandler(
                host=MONGO_URI,
                database_name=MONGO_DB,
                collection=MONGO_COLLECTION,
                # Pas besoin de username/password si déjà dans l'URI
                capped=True,
                capped_size=50 * 1024 * 1024,  # 50 MB
                buffer_size=100,
                buffer_periodical_flush_timing=10.0,
                # Utiliser une chaîne pour le niveau de flush
                buffer_early_flush_level='ERROR'  
            )
            mongo_handler.setLevel(log_level)
            logger.addHandler(mongo_handler)
            # Corriger la ligne de log avec une seule chaîne formatée
            logger.info(f"MongoDB logging activé vers {MONGO_DB}.{MONGO_COLLECTION}") 
        except Exception as e:
            logger.warning(f"Impossible d'initialiser le handler MongoDB: {e}. Le logging vers MongoDB est désactivé.")

    return logger