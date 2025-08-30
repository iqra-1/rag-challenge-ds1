# app.py
import gradio as gr
import os
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ===============================
# 1. Load & Preprocess Documents
# ===============================
uploaded_files = ["Attention is all you need.pdf", "EU AI Act.docx"]

docs = []
for file_path in uploaded_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Upload {file_path} to your Space.")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        docs.extend(loader.load())

# Normalize


def normalize_text(text):
    import re
    return re.sub(r"\s+", " ", text.lower().strip())


docs = [Document(page_content=normalize_text(d.page_content),
                 metadata=d.metadata) for d in docs]

# Split
text_splitter = NLTKTextSplitter(chunk_size=300, chunk_overlap=40)
chunked_documents = text_splitter.split_documents(docs)

# ===============================
# 2. Build Retriever
# ===============================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = FAISS.from_documents(chunked_documents, embedding_model)
faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 4})

bm25_retriever = BM25Retriever.from_documents(chunked_documents)
bm25_retriever.k = 2

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])

# ===============================
# 3. Load Generator (CPU-Safe)
# ===============================
model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# ===============================
# 4. Answer Function
# ===============================


def answer_query(query):
    # Retrieve
    retrieved_docs = retriever.get_relevant_documents(query)
    context = " ".join([d.page_content for d in retrieved_docs])

    # Prompt
    input_text = f"answer the question using the context. question: {query} context: {context}"

    try:
        result = generator(
            input_text,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
            truncation=True
        )
        answer = result[0]["generated_text"].strip()
        answer = answer.split("\n")[0].strip()
        return answer if answer else "Not found in context"
    except Exception as e:
        return "Error: " + str(e)


# ===============================
# 5. Gradio Interface
# ===============================
demo = gr.Interface(
    fn=answer_query,
    inputs="text",
    outputs="text",
    title="RAG Q&A System (CPU Version)",
    description="Ask a question about the uploaded documents. Model: FLAN-T5-Small (CPU-friendly).",
    examples=[
        ["Who proposed the Transformer model in 'Attention Is All You Need'?"],
        ["What is the key mechanism in the Transformer architecture?"],
        ["What AI systems are prohibited under the EU AI Act?"]
    ]
)

# Launch
if __name__ == "__main__":
    demo.launch()
