# Naive RAG YouTube

A Retrieval-Augmented Generation (RAG) application that allows you to chat with the content of YouTube videos using AI.

## 🎯 Overview

This project enables you to:
1. Enter a YouTube video URL
2. Automatically transcribe and index the video content
3. Ask questions about the video in natural language
4. Get AI-generated answers based on the actual video content

It combines several technologies:
- YouTube Transcript API for video transcription
- Sentence Transformers for text embedding
- Qdrant for vector storage and similarity search
- Groq for fast LLM inference
- Streamlit for the web interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   YouTube URL   │───▶│  Transcript API  │───▶│  Text Chunks     │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                              │
                                                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  User Question  │───▶│  Embedding Model │───▶│  Similarity      │
└─────────────────┘    └──────────────────┘    │    Search        │
                                                └──────────────────┘
                                                              │
                                                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Qdrant DB     │───▶│ Relevant Chunks  │───▶│  Groq LLM        │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                              │
                                                              ▼
                                                ┌──────────────────┐
                                                │   AI Answer      │
                                                └──────────────────┘
```

## 📁 Project Structure

```
naive-rag/
├── main.py                 # Main CLI entry point
├── streamlit_app.py        # Streamlit web interface
├── src/
│   ├── youtube.py          # YouTube URL handling and transcription
│   ├── embedding.py        # Text chunking and embedding
│   ├── qdrant.py           # Vector database operations
│   ├── retrieve.py         # Similarity search in Qdrant
│   ├── query.py            # LLM query generation
│   ├── grok.py             # Groq API client
│   ├── prompt.py           # Prompt templates
│   └── loggings.py         # Logging configuration
├── downloads/              # Temporary storage for transcripts
└── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Groq API key (free at [groq.com](https://groq.com))
- A Qdrant Cloud account (free tier available at [qdrant.tech](https://qdrant.tech))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/naive-rag-youtube.git
   cd naive-rag-youtube
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

### Running the Application

#### CLI Version
```bash
python main.py
```

#### Web Interface (Streamlit)
```bash
streamlit run streamlit_app.py
```

## 🧠 How It Works

### 1. Video Ingestion
1. User provides a YouTube URL
2. System extracts the video ID
3. Transcript is fetched using `youtube-transcript-api`
4. Text is split into chunks (700 chars with 100 overlap)
5. Each chunk is embedded using `sentence-transformers/all-mpnet-base-v2`
6. Embeddings + metadata are stored in Qdrant

### 2. Question Answering
1. User asks a question in the chat interface
2. Question is embedded using the same model
3. Similarity search finds top 5 relevant chunks in Qdrant
4. Chunks + conversation history are sent to Groq LLM
5. LLM generates a contextualized answer
6. Answer is displayed to the user

### 3. Key Features
- **Automatic Ingestion**: Videos are processed on first query
- **Conversation Context**: Maintains chat history for coherent responses
- **Model Selection**: Choose between different Groq models
- **Generation Parameters**: Adjustable temperature and max tokens
- **Multi-language Support**: Handles videos in different languages

## ⚙️ Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key for LLM access
- `QDRANT_URL`: Your Qdrant cluster URL
- `QDRANT_API_KEY`: Your Qdrant API key

### Available Models
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`
- `qwen/qwen3-32b`

### Adjustable Parameters
- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative)
- **Max Tokens**: Maximum length of generated response

## 🛠️ Development

### Main Modules

#### `src/youtube.py`
- Extracts video ID from YouTube URLs
- Saves transcripts to text files

#### `src/embedding.py`
- Splits text into manageable chunks
- Generates embeddings using Sentence Transformers
- Stores embeddings in Qdrant

#### `src/qdrant.py`
- Manages connection to Qdrant vector database
- Creates collections and indexes
- Handles upsert and search operations

#### `src/retrieve.py`
- Performs similarity search in Qdrant
- Filters results by video ID

#### `src/query.py`
- Formats prompts for the LLM
- Calls Groq API to generate responses

#### `src/grok.py`
- Wrapper for Groq API client
- Handles LLM inference

#### `src/prompt.py`
- Centralized prompt templates
- Structured prompts for better responses

### Logging
All modules use structured logging to `*.log` files for debugging and monitoring.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) for easy transcript retrieval
- [Sentence Transformers](https://www.sbert.net/) for powerful embeddings
- [Qdrant](https://qdrant.tech/) for the excellent vector database
- [Groq](https://groq.com/) for blazing-fast LLM inference
- [Streamlit](https://streamlit.io/) for the simple web framework

## 🚨 Limitations

- Only works with videos that have transcripts available
- Performance depends on the quality of the original transcript
- Free tiers of Qdrant and Groq have usage limits
- Large videos may take time to process initially

## 🔒 Privacy

- Video content is processed locally for transcription
- Only text chunks and embeddings are stored in Qdrant
- No personal data is collected or stored by the application
- API keys are stored locally in `.env` file

## ⚠️ Limitations

### YouTube IP Blocking
When deployed on cloud platforms (Streamlit Cloud, Render, etc.), YouTube often blocks requests for transcripts due to their restrictions on cloud server IPs.

**This is not a bug in the application but a limitation imposed by YouTube.**

**Workarounds:**
- Use videos that have manually added subtitles (more likely to be accessible)
- Run the application locally on your machine
- Consider alternative data sources for production deployments