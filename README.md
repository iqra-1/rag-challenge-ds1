# Data Scientist I – RAG Challenge Submission

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that answers questions from uploaded PDF and DOCX documents. It demonstrates core NLP and LLM engineering skills:
- Document preprocessing
- Hybrid retrieval
- Context-aware generation
- Evaluation
- Cloud deployment (bonus)

 **Live Demo (CPU Version)**: [https://iqb-my-rag-app.hf.space](https://iqb-my-rag-app.hf.space)

---

## 1. Data Preparation

### Steps
- **Document Loading**: Used `PyPDFLoader` and `Docx2txtLoader` to extract text from both file types.
- **Text Normalization**: Converted to lowercase and collapsed whitespace for consistent matching.
- **Chunking**: Split documents into 300-character chunks with 40-character overlap using `NLTKTextSplitter` to preserve sentence boundaries.

### Why?
- **Lowercasing**: Ensures case-insensitive retrieval (e.g., "Transformer" vs "transformer").
- **Sentence-aware splitting**: Prevents breaking key facts across chunks.
- **Overlap**: Maintains context continuity.
- **Small chunks**: Balance between precision and coverage.

This ensures the system can find and use relevant snippets effectively.

---

## 2. Retrieval Component

### Method
- **Ensemble Retriever** combining:
  - **FAISS** + `all-mpnet-base-v2` embeddings (semantic search)
  - **BM25** (keyword search)
- Weights: `[0.3, 0.7]` → favors semantic but keeps keyword recall.

### Retrieval Demo
For a query like `"Who proposed the Transformer model?"`, the system retrieves top-4 relevant chunks showing:
- Source document
- Page number
- Context preview

This proves the retriever finds accurate, traceable information.

---

## 3. Generation Component

### Two Models Used

#### CPU Version: `google/flan-t5-small`
- Used for **deployment** on Hugging Face Spaces (CPU tier)
- Lightweight, works without GPU
- Lower accuracy, especially on extraction tasks
- **Live at**: [https://iqb-my-rag-app.hf.space](https://iqb-my-rag-app.hf.space)

#### GPU Version: `mistralai/Mistral-7B-Instruct`
- Used for **evaluation and testing**
- Instruction-tuned → follows prompts better
- Higher accuracy (3/3 correct) and less hallucination
- Requires GPU and HF token

### Prompt Design
Used model-specific templates:
- **Mistral**: Chat template (`<|user|>`, `<|assistant|>`) for strict grounding
- **FLAN-T5**: Simple `question: ... context: ...` format

This ensures the model uses only retrieved context.

---

## 4. Evaluation

### Metrics
1. **Answer Accuracy**: % of queries where prediction contains expected answer (flexible substring match).
2. **Retrieval Recall@4**: % of queries where correct answer appears in top 4 retrieved chunks.

### Test Queries
- "Who proposed the Transformer model in 'Attention Is All You Need'?"
- "What is the key mechanism used in the Transformer architecture?"
- "What AI systems are prohibited under the EU AI Act?"

### Results
| Model | Answer Accuracy | Retrieval Recall@4 |
|------|------------------|--------------------|
| FLAN-T5 (CPU) | ~1/3 | 0.67 |
| Mistral (GPU) | 3/3 (100%) | 1.0 |

 The GPU model achieved **perfect accuracy** and **no hallucinations**
 The CPU model works but is less reliable

---

## 5. Bonus: Cloud Deployment

### Platform
Deployed on **Hugging Face Spaces** with Gradio UI.

### Link
-  [https://iqb-my-rag-app.hf.space](https://iqb-my-rag-app.hf.space)

### Why This Counts
- Hugging Face Spaces runs on AWS and Google Cloud infrastructure
- Public, interactive, and self-contained
- Demonstrates full-stack ML engineering
- Meets the "cloud deployment" bonus requirement

### Trade-off
- Used **FLAN-T5** instead of Mistral due to CPU-only deployment
- Lower accuracy, but shows functional pipeline

---

## How to Run Locally

```bash
# Clone repo
git clone https://github.com/your-username/rag-challenge-ds1.git
cd rag-challenge-ds1

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook rag_pipeline.ipynb
