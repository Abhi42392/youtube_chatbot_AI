import os
from functools import partial
from urllib.parse import urlparse, parse_qs

import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import CrossEncoder

load_dotenv()

# ----- Config -----
INDEX_NAME = "youtube-transcript"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384                      # must match all-MiniLM-L6-v2's output size
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVE_TOP_K = 8                   # broad candidate set from hybrid search
RERANK_TOP_N = 4                     # keep only the best few after cross-encoder
ALPHA = 0.7                          # hybrid weighting: 0.7 dense / 0.3 sparse

# ----- One-time setup of models/clients (shared across reruns) -----
# temperature low (0.3) because this is factual Q&A grounded in context,
# not creative writing — we want faithful, consistent answers.
llm = HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V3.2", temperature=0.3)
model = ChatHuggingFace(llm=llm)

# Cross-encoder: scores (query, doc) PAIRS together for accurate reranking.
# Slower than the bi-encoder embeddings, so we only run it on the top-k candidates.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def get_video_id(url: str) -> str | None:
    """Extract the video id from a standard or short YouTube URL."""
    parsed = urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed.query).get("v", [None])[0]
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    return None


def rerank(query: str, retriever) -> list[str]:
    """Retrieve candidates via hybrid search, then rerank them with the
    cross-encoder and return the top-N chunk texts (plain strings)."""
    candidates = retriever.invoke(query)
    docs = [d.page_content for d in candidates if d is not None]
    if not docs:
        return []

    # Cross-encoder scores each (query, doc) pair; higher = more relevant.
    pairs = [(query, doc) for doc in docs]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    # Return ONLY the text of the top-N docs (this is the bug fix — the old
    # code passed (doc, score) tuples downstream and tried to join them).
    return [doc for doc, _ in ranked[:RERANK_TOP_N]]


def format_context(docs: list[str]) -> str:
    """Join the reranked chunk texts into a single context block."""
    return "\n\n".join(docs)


# ----- UI -----
url = st.text_input("Enter the YouTube video URL")

if url:
    video_id = get_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid URL.")
        st.stop()

    # Only (re)build the chain when the video changes.
    if "chain" not in st.session_state or st.session_state.get("video_id") != video_id:
        try:
            with st.spinner("Processing video transcript..."):
                # 1. Fetch transcript
                transcript_arr = YouTubeTranscriptApi().fetch(video_id)
                transcript = " ".join(t.text for t in transcript_arr)

                # 2. Split into overlapping chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                corpus = [d.page_content for d in splitter.create_documents([transcript])]

                # 3. Dense (semantic) + sparse (BM25 keyword) encoders
                embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                bm25_encoder = BM25Encoder()
                bm25_encoder.fit(corpus)

                # 4. Batch-embed all chunks (faster than one at a time)
                all_dense = embeddings.embed_documents(corpus)

                # 5. Create the index once if it doesn't exist
                if INDEX_NAME not in pc.list_indexes().names():
                    pc.create_index(
                        name=INDEX_NAME,
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                        metric="dotproduct",
                        dimension=EMBED_DIM,
                    )
                index = pc.Index(INDEX_NAME)

                # 6. Upsert into a per-video NAMESPACE so queries never pull
                #    chunks from a different video (fixes cross-video bleed).
                vectors = [
                    {
                        "id": f"{video_id}_{i}",
                        "values": all_dense[i],
                        "sparse_values": bm25_encoder.encode_documents(text),
                        "metadata": {"text": text},
                    }
                    for i, text in enumerate(corpus)
                ]
                for i in range(0, len(vectors), 100):
                    index.upsert(vectors=vectors[i:i + 100], namespace=video_id)

                # 7. Hybrid retriever scoped to this video's namespace
                retriever = PineconeHybridSearchRetriever(
                    index=index,
                    embeddings=embeddings,
                    sparse_encoder=bm25_encoder,
                    text_key="text",
                    alpha=ALPHA,
                    top_k=RETRIEVE_TOP_K,
                    namespace=video_id,
                )

                # 8. Prompt: force the model to answer only from context.
                prompt = PromptTemplate(
                    input_variables=["context", "query"],
                    template=(
                        "You are a helpful chatbot. Answer the question using the "
                        "provided context only. If the context is insufficient, say "
                        "you don't know.\n"
                        "Context: {context}\n"
                        "Question: {query}\n"
                    ),
                )

                # 9. RAG chain: retrieve+rerank -> format -> prompt -> LLM -> text
                #    partial() binds this specific retriever into rerank(),
                #    so we don't depend on a fragile global.
                parallel_chain = RunnableParallel(
                    context=RunnableLambda(partial(rerank, retriever=retriever))
                    | RunnableLambda(format_context),
                    query=RunnablePassthrough(),
                )
                final_chain = parallel_chain | prompt | model | StrOutputParser()

                st.session_state.chain = final_chain
                st.session_state.video_id = video_id

            st.success("Video processed! Ask your questions below.")

        except Exception as e:
            st.error(f"Error processing video: {e}")
            st.stop()

    # Query using the cached chain (no reprocessing).
    query = st.text_input("Ask a question about the video")
    if query:
        with st.spinner("Thinking..."):
            st.write(st.session_state.chain.invoke(query))