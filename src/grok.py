# grok.py

from groq import Groq
import os
from .loggings import configure_logging

logger = configure_logging(log_file="grok.log", logger_name="__grok__")

DEFAULT_MODEL = "openai/gpt-oss-120b" 

def get_grok_client() -> Groq:
    """
    Returns the Groq client.

    Returns:
        Groq: The Groq client.
    """
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        logger.error("La variable d'environnement GROK_API_KEY n'est pas définie.")
        raise ValueError("GROK_API_KEY is not set.")
        
    client = Groq(api_key=api_key)
    logger.info("Groq client created")
    return client

def generate_answer_with_grok(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 500, temperature: float = 0.2) -> str:
    """
    Generates an answer using the Groq API.

    Args:
        prompt (str): The prompt to send to the model.
        model (str): The model name to use (e.g., 'openai/gpt-oss-120b').
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The sampling temperature.

    Returns:
        str: The generated answer.
    """
    logger.info(f"Generating answer with Groq model '{model}'")
    client = get_grok_client()
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = chat_completion.choices[0].message.content.strip()
        logger.info("Answer generated successfully.")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer with Groq: {e}")
        return "Désolé, une erreur s'est produite lors de la génération de la réponse avec Grok."
