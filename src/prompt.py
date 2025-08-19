
# Prompt principal pour le RAG
RAG_PROMPT_TEMPLATE = """Tu es un assistant expert spécialisé dans l'analyse de contenu vidéo YouTube. Ta tâche est de répondre de manière précise, structurée et utile aux questions en te basant uniquement sur le contexte fourni.

**Instructions importantes :**
- Utilise UNIQUEMENT les informations du contexte fourni
- Si le contexte ne contient pas d'information pertinente, dis-le clairement
- Structure ta réponse de manière logique (introduction, points principaux, conclusion si pertinent)
- Sois concis mais complet
- Si demandé, cite des parties spécifiques du contexte

**Contexte de la conversation :**
{conversation_context}

**Contexte extrait de la vidéo :**
{relevant_context}

**Question actuelle :**
{question}

**Réponse :**
"""

# Prompt pour les questions sans contexte pertinent
NO_CONTEXT_PROMPT = """Tu es un assistant utile. Malheureusement, je n'ai pas trouvé d'information pertinente dans la vidéo pour répondre à cette question.

Question : {question}

Réponse : Je ne trouve pas d'information pertinente dans les transcriptions fournies pour répondre à votre question. Veuillez poser une question liée au contenu de la vidéo."""

# Prompt pour l'analyse de la pertinence (optionnel - pour améliorations futures)
RELEVANCE_PROMPT_TEMPLATE = """Évalue si le contexte suivant est pertinent pour répondre à la question.

Contexte : {context}
Question : {question}

Réponds par 'OUI' ou 'NON' uniquement."""

def format_rag_prompt(question: str, relevant_chunks: list, conversation_history: list = None) -> str:
    """
    Formate le prompt complet pour le RAG.
    
    Args:
        question (str): La question utilisateur
        relevant_chunks (list): Liste des chunks pertinents
        conversation_history (list): Historique de conversation
        
    Returns:
        str: Le prompt formaté
    """
    # Construire le contexte pertinent
    relevant_context = "\n\n---\n\n".join([chunk.get("text", "") for chunk in relevant_chunks])
    
    # Construire le contexte de conversation
    if conversation_history and len(conversation_history) > 0:
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history[-3:]  # Garder les 3 derniers échanges
        ])
    else:
        conversation_context = "Aucun historique de conversation."
    
    return RAG_PROMPT_TEMPLATE.format(
        conversation_context=conversation_context,
        relevant_context=relevant_context,
        question=question
    )

def format_no_context_prompt(question: str) -> str:
    """Formate le prompt quand aucun contexte n'est trouvé."""
    return NO_CONTEXT_PROMPT.format(question=question)