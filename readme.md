# Generative AI & LLMs: Course Assignments

This repository contains the implementation and documentation for two major projects focused on **Retrieval-Augmented Generation (RAG)** and **LLM Fine-Tuning**.

---

## 🚀 Assignment 1: RAG for Domain-Specific Question Answering

### Objective
Design and implement a **Retrieval-Augmented Generation (RAG)** system capable of answering user queries using a domain-specific knowledge base to mitigate hallucinations and lack of domain knowledge.

### 📂 Problem Statement
LLMs often generate incorrect responses when specialized knowledge is required. This project implements a RAG-based application to retrieve relevant documents and inject them into the prompt context.

**Selected Domains (Choose one):**
*   Healthcare / Legal / University / Research / Finance / Technical / Customer Support

### 🛠 System Components
*   **Document Ingestion:** Preprocessing of raw domain data.
*   **Chunking Strategy:** Evaluation of various chunk sizes for optimal context.
*   **Embedding Generation:** Comparative use of different models (e.g., HuggingFace) and dimensions.
*   **Vector Databases:** Implementation using at least **two** vector stores (e.g., Pinecone, Milvus, ChromaDB).
*   **Prompt Engineering:** Semantic search and ranking strategies.
*   **LLM Generation:** Inference using models like **LLaMA-3.1, Gemma-3,** or GPT-style open models.

### 📤 Deliverables
1.  **Source Code:** Modular design for ingestion, retrieval, and generation.
2.  **Experimental Report:** An 8–10 page analysis of configurations.
3.  **Comparative Evaluation:** Metrics comparison between two different system setups.
4.  **Demonstration:** Screenshots and sample outputs of the system in action.

---

## ⚖️ Assignment 2: Fine-Tuning an LLM Using Custom Dataset

### Objective
Understand how fine-tuning improves LLM performance for specialized tasks by adapting pretrained models to domain-specific datasets.

### 📂 Problem Statement
Adapting general-purpose LLMs (LLaMA, Mistral, Gemma) to perform specialized tasks such as:
*   Instruction-following
*   Code generation
*   Domain-specific summarization
*   Structured data-to-text

### 📊 Dataset Requirements
*   **Size:** 500 – 2,000 examples.
*   **Format:** Paired "Input Prompt" and "Expected Output."
*   **Sources:** Synthetic (LLM-generated), Public, or Custom Curated.

### 🧠 Technical Approach
*   **Model Options:** LLaMA, Mistral, Gemma, or other open-source LLMs.
*   **Methods:**
    *   **PEFT (Parameter-Efficient Fine-Tuning)**
    *   **LoRA / QLoRA**
    *   HuggingFace Transformers Library

### 📤 Deliverables
1.  **Training Pipeline:** Complete source code for preprocessing and fine-tuning.
2.  **Dataset Documentation:** Full description of the curated dataset.
3.  **Experimental Report (8–10 pages):** 
    *   Methodology and Training configuration.
    *   Evaluation results (Accuracy, BLEU/ROUGE).
4.  **Comparative Analysis:** Performance improvement of the **Fine-Tuned model vs. Base model.**
5.  **Demonstration:** Screenshots showing the difference in output quality.

---


# Install dependencies
pip install -r requirements.txt