import streamlit as st
import time
import os

from dotenv import load_dotenv
from openai import OpenAI
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import DistanceStrategy

# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()

# ==========================================
# DEEPINFRA CLIENT
# ==========================================

import streamlit as st
client = OpenAI(
    api_key=st.secrets["API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai"
)

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Technical Support AI Bot",
    layout="wide"
)

st.title("Technical Support AI Bot")

# ==========================================
# LOAD PDF + BUILD VECTORSTORE
# ==========================================

@st.cache_resource
def build_vectorstore():
    loader = PyPDFLoader("API Documentation Partial.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    sample_embedding = embedding_model.embed_query("test")
    dimension = len(sample_embedding)

    index = faiss.IndexFlatIP(dimension)

    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )

    vectorstore.add_documents(chunks)
    return vectorstore


def semantic_search(user_query, vectorstore, k=3):
    results = vectorstore.similarity_search(user_query, k=k)
    return results

# ==========================================
# GENERATE RAG ANSWER
# ==========================================

def generate_rag_answer(query, vectorstore):

    # ==========================================
    # RETRIEVE RELEVANT DOCUMENTS
    # ==========================================

    retrieved_docs = semantic_search(
        query,
        vectorstore,
        k=3
    )

    # ==========================================
    # BUILD CONTEXT
    # ==========================================

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # ==========================================
    # SYSTEM PROMPT
    # ==========================================

    system_prompt = """
You are a Senior Upwork API Consultant.

Rules:
1. Answer ONLY from the provided context.
2. Do NOT hallucinate or make up information.
3. If the answer is not available in the context,
   respond with:
   "I'm sorry, but the provided documentation does not contain that information."
4. Keep answers concise and technical.
5. If it is not relavant to the question ignore
"""

    # ==========================================
    # USER PROMPT
    # ==========================================

    user_prompt = f"""
Context:
{context}

Question:
{query}
"""

    # ==========================================
    # LATENCY START
    # ==========================================

    start_time = time.time()

    # ==========================================
    # GENERATE RESPONSE
    # ==========================================

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_tokens=300,
        temperature=0.3
    )

    # ==========================================
    # LATENCY END
    # ==========================================

    end_time = time.time()

    latency = round(end_time - start_time, 2)

    # ==========================================
    # FINAL ANSWER
    # ==========================================

    answer = response.choices[0].message.content

    return answer, retrieved_docs, latency

# ==========================================
# BUILD VECTORSTORE
# ==========================================

with st.spinner("Loading PDF and building vector DB..."):
    vectorstore = build_vectorstore()

st.success("Vector database ready!")

# ==========================================
# USER INPUT
# ==========================================

query = st.text_input(
    "Question"
)

# ==========================================
# GENERATE RESPONSE
# ==========================================

if query:

    with st.spinner("Generating answer"):

        answer, sources, latency = generate_rag_answer(
            query,
            vectorstore
        )

    # ==========================================
    # ANSWER
    # ==========================================

    st.subheader("AI Generated Answer")

    st.write(answer)

    # ==========================================
    # LATENCY
    # ==========================================

    st.subheader("Latency")

    st.write(f"{latency} seconds")

    # ==========================================
    # SOURCES
    # ==========================================

    st.subheader("Sources")

    for i, doc in enumerate(sources, 1):

        with st.expander(f"Source {i}"):

            st.write(doc.page_content)

            st.write("---")

            st.write(
                f"Page: {doc.metadata.get('page')}"
            )