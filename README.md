# Technical Support AI Bot

## Overview

This project implements a Technical Support AI Bot using Streamlit and Retrieval-Augmented Generation (RAG) techniques. It leverages LLMs (Large Language Models) and vector search (FAISS) to answer technical queries based on provided documentation.

---

## Project Structure

- **app.py**: Streamlit web application for user interaction with the AI bot.
- **main.ipynb**: Jupyter notebook containing the data processing pipeline, including PDF loading, text chunking, embedding, and vector store creation.
- **requirements.txt**: Lists all required Python libraries.
- **pyproject.toml**: Project metadata and dependencies for reproducibility.
- **DB/faiss_index/**: Directory containing the FAISS vector index files (`index.faiss`, `index.pkl`) for fast semantic search.
- **Technical Summary.md**: Insights, challenges, and technical notes about the project.

---

## Key Features

- **Document Loading & Chunking**: Loads PDF documentation, splits text into overlapping chunks to preserve context.
- **Embeddings**: Uses HuggingFace sentence-transformers to embed document chunks.
- **Vector Search**: Stores embeddings in a FAISS index for efficient similarity search.
- **RAG Pipeline**: Retrieves relevant context and generates answers using an LLM (via DeepInfra/OpenAI API).
- **Streamlit UI**: User-friendly web interface for querying the bot.

---

## Setup & Usage

1. **Install dependencies**  
    Use the provided requirements.txt or pyproject.toml to install all dependencies:
    ```
    pip install -r requirements.txt
    ```
    or with a tool like `uv` or `pip` for pyproject.toml.

2. **Set Environment Variables**  
    Create a `.env` file with your API keys:
    ```
    API_KEY=your_deepinfra_or_openai_key
    MODEL_NAME=your_model_name
    ```

3. **Run the Application**  
    Launch the Streamlit app:
    ```
    streamlit run app.py
    ```

4. **Interact**  
    Open the provided local URL in your browser and start asking technical questions.

---

## Notable Libraries Used

- `streamlit` - Web UI
- `langchain`, `langchain_community` - Document processing and RAG
- `faiss-cpu` - Vector search
- `sentence-transformers` - Embeddings
- `openai` - LLM API
- `python-dotenv` - Environment variable management

---

## Technical Notes

- **Chunk Overlap**: Overlap is used when splitting text to ensure context is preserved across chunk boundaries, which is crucial for technical content.
- **Caching**: Streamlit's `@st.cache_resource` is used to improve latency.
- **API Integration**: The app uses DeepInfra's OpenAI-compatible API for LLM responses.

---

## Troubleshooting

- Ensure all dependencies are installed and the correct Python version (>=3.12) is used.
- Make sure your `.env` file is present and contains valid API credentials.
- If you encounter vector index errors, ensure the `DB/faiss_index/` directory contains both `index.faiss` and `index.pkl`.

---