# query.py

from src.loggings import configure_logging
from src.grok import generate_answer_with_grok 
from src.prompt import format_rag_prompt, format_no_context_prompt 
from typing import List, Dict

logger = configure_logging(log_file="query.log", logger_name="__query__")

def build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Construit un prompt à partir de la question et des chunks récupérés.
    
    Args:
        question (str): La question de l'utilisateur.
        chunks (List[Dict]): La liste des chunks récupérés, avec leurs 'text'.

    Returns:
        str: Le prompt formaté.
    """
    # Joindre les textes des chunks avec des sauts de ligne
    context_text = "\n\n---\n\n".join([chunk["text"] for chunk in chunks])
    
    # Créer le prompt. Tu peux ajuster ce template selon tes besoins.
    prompt = f"""
            Tu es un assistant utile. Réponds à la question en utilisant uniquement le contexte fourni ci-dessous.
            Si le contexte ne contient pas l'information nécessaire pour répondre, dis simplement: "Je ne trouve pas d'information pertinente dans les transcriptions fournies."

            Contexte:
            {context_text}

            Question:
            {question}

            Réponse:
    """
    logger.debug("Prompt construit avec succès.")
    return prompt.strip()

def answer_question_with_grok(question: str, chunks: List[Dict], model: str = "llama3-8b-8192", max_tokens: int = 500, temperature: float = 0.2, conversation_history: List[Dict] = None) -> str:
    """
    Pipeline complet : construit le prompt, appelle Grok et retourne la réponse.
    
    Args:
        question (str): La question de l'utilisateur.
        chunks (List[Dict]): Les chunks récupérés par retrieve.py.
        model (str): Le modèle Groq à utiliser.
        max_tokens (int): Nombre max de tokens pour la réponse.
        temperature (float): Température pour la génération.
        conversation_history (List[Dict]): Historique de conversation optionnel.

    Returns:
        str: La réponse générée par le LLM.
    """
    logger.info(f"Construction du prompt pour la question: '{question}'")
    
    # Vérifier s'il y a des chunks pertinents
    if not chunks or len(chunks) == 0:
        logger.info("Aucun chunk pertinent fourni, utilisation du prompt sans contexte")
        prompt = format_no_context_prompt(question)
    else:
        # Utiliser le prompt RAG structuré
        prompt = format_rag_prompt(question, chunks, conversation_history)
    
    logger.debug(f"Prompt construit (longueur: {len(prompt)} caractères)")
    logger.info("Appel à Grok pour générer la réponse...")
    
    answer = generate_answer_with_grok(prompt, model=model, max_tokens=max_tokens, temperature=temperature)
    
    logger.info("Réponse générée avec succès.")
    return answer