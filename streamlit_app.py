import streamlit as st
import os
import zipfile

st.title("Fusion RAG Chatbot")

# File upload widget
uploaded_file = st.file_uploader("Upload FAISS index zip", type=["zip"])

if uploaded_file is not None:
    # Save uploaded file
    with open("faiss_index.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Unzip it
    with zipfile.ZipFile("faiss_index.zip", "r") as zip_ref:
        zip_ref.extractall("content/")

    st.success("FAISS index extracted successfully!")


faiss_index_folder = "/content/faiss_index"
from huggingface_hub import login

login("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")

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

FAISS_FOLDER = faiss_index_folder  # your FAISS folder
EMBED_MODEL = "sentence-transformers/sentence-t5-large"  # same model used when creating FAISS
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # gated; needs HF token with access to gated models
HF_TOKEN = os.getenv("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")  # set in env. Make sure this token has access to gated models if using a gated LLM.

# ====== Load FAISS ======
# The LangChainDeprecationWarning is expected as HuggingFaceEmbeddings has moved to a new package.
# For this example, using the current version is acceptable.
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
db = FAISS.load_local(FAISS_FOLDER, embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# ====== Load LLaMA ======
# The error "403 Forbidden" indicates that the Hugging Face token does not have permission to access
# the gated 'meta-llama/Llama-2-7b-chat-hf' model.
# To resolve this, ensure your Hugging Face token has access to gated models and update the 'HF_TOKEN'
# secret in Colab, then restart the runtime.
# Alternatively, you can use a non-gated model for demonstration purposes, e.g., "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    resume_download=True
)
# ====== Fusion RAG Components ======
def generate_queries(original_query: str):
    """Expand the original query into multiple related queries."""
    return [
        original_query,
        f"Explain in detail: {original_query}",
        f"What are the advantages of {original_query}?",
        f"What are the challenges or limitations of {original_query}?",
        f"Give a real-world application of {original_query}"
    ]

def reciprocal_rank_fusion(results_dict, k=60):
    """Fuse rankings from multiple queries."""
    fused_scores = {}
    for query, docs in results_dict.items():
        for rank, doc in enumerate(docs):
            fused_scores[doc.page_content] = fused_scores.get(doc.page_content, 0) + 1 / (rank + k)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

# ====== Fusion RAG Answer ======
def fusion_rag_answer(query):
    expanded_queries = generate_queries(query)
    all_results = {}

    # Step 1: Retrieve docs for each query
    for q in expanded_queries:
        docs = retriever.get_relevant_documents(q)
        all_results[q] = docs

    # Step 2: Fuse results
    reranked = reciprocal_rank_fusion(all_results)

    # Step 3: Build context (top 5 passages)
    top_passages = [doc for doc, _ in reranked[:5]]
    context = "\n\n".join(top_passages)

    # Step 4: Create prompt for LLaMA
    prompt = f"""
   You are a helpful assistant. Use only the information provided in the context to answer the user’s question.

Instructions:
- Base your answer strictly on the context. Do not use outside knowledge.
- If the context does not fully answer the question, say: "The context does not provide this information."
- Summarize clearly and concisely.
- Highlight key details directly from the context.
- If multiple pieces of evidence are relevant, integrate them into a well-structured response.
- Prefer short paragraphs or bullet points for clarity

    Context:
    {context}

    Question: {query}
    Answer:
    """

    # Step 5: Generate answer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====== Chat Loop ======
print("Fusion RAG Chatbot Ready. Type 'exit' to quit.")
while True:
    q = input("\nYou: ")
    if q.lower() in ["exit", "quit"]:
        break
    print("\nBot:", fusion_rag_answer(q))
