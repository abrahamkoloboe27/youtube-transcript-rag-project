# Dockerfile

# Étape 1 : Construire les dépendances
FROM python:3.12-slim AS builder

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Créer un utilisateur non-root pour la construction
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copier les fichiers de spécification des dépendances
COPY --chown=app:app pyproject.toml uv.lock* ./

# Créer un environnement virtuel et installer les dépendances
# Utiliser le cache de uv pour accélérer les builds
RUN --mount=type=cache,target=/home/app/.cache/uv \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install -- locked --no-install-project --no-editable

# Étape 2 : Copier le code et synchroniser le projet
# (Cette étape pourrait être combinée avec la précédente si le code est petit)
FROM python:3.12-slim AS builder-with-code

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Créer l'utilisateur
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copier les fichiers de spécification ET le code source
# Assure-toi que ton .dockerignore est configuré pour exclure les fichiers inutiles
COPY --chown=app:app . .

# Synchroniser le projet (installe les dépendances ET le projet en mode éditable)
# Utiliser le cache de uv
RUN --mount=type=cache,target=/home/app/.cache/uv \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install -- locked --editable


# Étape 3 : Image de production
FROM python:3.12-slim

# Installer les dépendances système nécessaires (si besoin, par exemple pour sentence-transformers)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Créer l'utilisateur non-root
RUN useradd --create-home --shell /bin/bash app
USER app

# Définir le répertoire de travail
WORKDIR /home/app

# Copier l'environnement virtuel de l'étape de construction
COPY --from=builder-with-code --chown=app:app /home/app/.venv /home/app/.venv

# Copier le code source de l'application
# Si tu as copié le code dans builder-with-code, cette étape est redondante.
# Sinon, copie-le ici. Je suppose qu'il est déjà dans builder-with-code.
# COPY --chown=app:app . .

# S'assurer que l'environnement virtuel est activé
ENV PATH="/home/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/app/.venv"

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Définir la commande par défaut
# Utiliser uv pour exécuter Streamlit
# Assure-toi que streamlit_app.py est à la racine du dossier copié
CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
