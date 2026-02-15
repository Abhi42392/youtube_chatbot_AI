from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import streamlit as st
import os

load_dotenv()

llm = HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V3.2", temperature=1.8)
model = ChatHuggingFace(llm=llm)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed.query).get("v", [None])[0]
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    return None


def get_context(retrieved_docs):
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

def re_ranking_retriever(query):
    #initial records retrieved by retriever
    intially_retrieved_docs=retriever.invoke(query)
    # print("Initially retrieved docs")
    # for doc in intially_retrieved_docs:
    #     print(doc.page_content)
    #     print("----")
    # print()
    #re-ranking the docs using cross encoder
    pairs=[(query,doc.page_content) for doc in intially_retrieved_docs]

    scores=cross_encoder.predict(pairs)

    re_ranked_docs=sorted(
        zip(intially_retrieved_docs,scores),
        key=lambda x:x[1],
        reverse=True
    )
    # print("Re-ranked docs")
    # for doc,score in re_ranked_docs:
    #     print(f"Score: {score}")
    #     print(doc.page_content)
    #     print("----")
    re_ranked_docs=[doc for doc,score in re_ranked_docs]
    
    # print()

    return re_ranked_docs



url = st.text_input("Enter the YouTube video URL")

if url:
    video_id = get_video_id(url)

    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid URL.")
        st.stop()

    # Only process once per video — skip if already done
    if "chain" not in st.session_state or st.session_state.get("video_id") != video_id:
        try:
            with st.spinner("Processing video transcript..."):
                # 1. Fetch transcript
                ytt_api = YouTubeTranscriptApi()
                transcript_arr = ytt_api.fetch(video_id)
                transcript = " ".join([t.text for t in transcript_arr])

                # 2. Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splitted_text = text_splitter.create_documents([transcript])
                corpus = [doc.page_content for doc in splitted_text]

                # 3. Setup embeddings and BM25
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                bm25_encoder = BM25Encoder()
                bm25_encoder.fit(corpus)

                # 4. Batch embed all chunks at once (faster than one by one)
                all_dense = embeddings.embed_documents(corpus)

                # 5. Setup Pinecone index
                index_name = "youtube-transcript"
                if index_name not in pc.list_indexes().names():
                    pc.create_index(
                        name=index_name,
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                        metric="dotproduct",
                        dimension=384,
                    )
                index = pc.Index(index_name)

                # 6. Batch upsert (instead of one at a time)
                vectors = []
                for i, text in enumerate(corpus):
                    vectors.append({
                        "id": f"{video_id}_{i}",
                        "values": all_dense[i],
                        "sparse_values": bm25_encoder.encode_documents(text),
                        "metadata": {"text": text},
                    })
                for i in range(0, len(vectors), 100):
                    index.upsert(vectors=vectors[i:i + 100])

                # 7. Setup retriever
                retriever = PineconeHybridSearchRetriever(
                    index=index,
                    embeddings=embeddings,
                    sparse_encoder=bm25_encoder,
                    text_key="text",
                    alpha=0.7,
                    top_k=5
                )

                # 8. Build RAG chain
                prompt = PromptTemplate(
                    input_variables=["context", "query"],
                    template=(
                        "You are a helpful chatbot. Answer the question based on the provided context only. "
                        "If the context is insufficient just say you dont know.\n"
                        "Context: {context}\n"
                        "Question: {query}\n"
                    ),
                )

                

                parallel_chain = RunnableParallel(
                    context=RunnableLambda(re_ranking_retriever) | RunnableLambda(get_context),
                    query=RunnablePassthrough(),
                )

                final_chain = parallel_chain | prompt | model | StrOutputParser()

                # 9. Cache everything in session state
                st.session_state.chain = final_chain
                st.session_state.video_id = video_id

            st.success("Video processed! Ask your questions below.")

        except Exception as e:
            st.error(f"Error processing video: {e}")
            st.stop()

    # Query using cached chain (fast — no reprocessing)
    query = st.text_input("Ask a question about the video")
    # query="What is this video about"
    if query:
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke(query)
            st.write(result)