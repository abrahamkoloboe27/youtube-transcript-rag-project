# Dockerfile
# Étape 1 : Image de construction avec le code source
FROM python:3.12-slim AS builder

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Créer un utilisateur non-root pour la construction
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copier les fichiers de spécification ET le code source
# Assure-toi que ton .dockerignore est configuré pour exclure les fichiers inutiles
COPY --chown=app:app . .

# Créer un environnement virtuel et synchroniser le projet
# Utiliser le cache de uv pour accélérer les builds
RUN --mount=type=cache,target=/home/app/.cache/uv \
    uv venv && \
    uv sync --locked

# Étape 2 : Image de production
FROM python:3.12-slim



# Créer l'utilisateur non-root
RUN useradd --create-home --shell /bin/bash app
USER app

# Définir le répertoire de travail
WORKDIR /home/app

# Copier l'environnement virtuel de l'étape de construction
COPY --from=builder --chown=app:app /home/app/.venv /home/app/.venv

# Copier le code source de l'application
COPY --from=builder --chown=app:app /home/app /home/app

# S'assurer que l'environnement virtuel est activé
ENV PATH="/home/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/app/.venv"

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Définir la commande par défaut
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]