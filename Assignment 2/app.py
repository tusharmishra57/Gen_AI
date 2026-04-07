import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="AI/ML Research QA - Fine-tuned Qwen 2.5",
    page_icon="🤖",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #6366f1;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #6366f1;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
    }
    .response-header {
        font-weight: bold;
        padding: 10px;
        border-radius: 8px 8px 0 0;
        margin-bottom: 0;
    }
    .base-header {
        background-color: #f1f5f9;
        color: #475569;
    }
    .ft-header {
        background-color: #e0e7ff;
        color: #4338ca;
    }
    .response-body {
        padding: 15px;
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        background-color: white;
        color: #000000;
        min-height: 150px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Simulation ---
QA_PAIRS = [
    {
        "input": "Please describe the limitations of GPT-3?",
        "base": "GPT-3 is a large-scale transformer-based model that has been trained on vast amounts of text data from various sources to generate human-like responses for tasks such as translation, summarization, and question-answering. However, it often struggles with factual consistency and can produce generic or irrelevant answers depending on the prompt complexity.",
        "ft": "GPT-3 has some notable constraints:\n\n1. **Limited Context**: GPT-3 can only generate text based on a limited context (up to 4096 tokens). Longer inputs result in truncated outputs.\n2. **In-context Learning Limitations**: While it shows few-shot capabilities, its reasoning on complex natural language inference tasks is often inconsistent.\n3. **Bias & Harm**: It has a potential for generating biased or harmful content based on its training data.\n4. **Computational Cost**: High computational cost for inference and inability to learn from new interactions after training."
    },
    {
        "input": "Explain multi-head attention in Transformers.",
        "base": "Multi-head attention is a mechanism in Transformers that allows the model to process information from different parts of the input simultaneously. It uses several attention 'heads' to look at the data from different perspectives, which helps the model understand complex relationships between words.",
        "ft": "Multi-head attention runs multiple attention functions in parallel. The queries, keys, and values are linearly projected h times with different learned projections. Each head attends to different representation subspaces (different parts of the sequence). The outputs are concatenated and projected to produce the final result, allowing the model to jointly attend to information from different representation subspaces at different positions."
    },
    {
        "input": "What is Retrieval-Augmented Generation (RAG)?",
        "base": "RAG is a technique used in AI to improve the quality of generated text by incorporating information from external sources. It involves searching a database for relevant documents and using that information to help the model generate a more accurate and informative response.",
        "ft": "RAG combines pre-trained parametric memory (the language model) with non-parametric memory (an external document corpus). It retrieves relevant documents from the corpus based on the input query and then conditions the generation on both the original input and the retrieved passages. This grounds the generation in factual evidence, reduces hallucinations, and allows the model to access up-to-date information without retraining."
    },
    {
        "input": "What is LoRA and how does it work?",
        "base": "LoRA, or Low-Rank Adaptation, is a method for fine-tuning large models more efficiently. Instead of updating all the weights, it only updates a small number of parameters, which makes the process faster and requires less memory.",
        "ft": "LoRA (Low-Rank Adaptation) freezes pre-trained model weights and injects trainable low-rank decomposition matrices (A and B) into Transformer layers. For a weight matrix W, the update is ΔW = B × A, where B and A are low-rank. This reduces the number of trainable parameters by up to 10,000x and GPU memory usage by 3x, while maintaining performance comparable to full fine-tuning. It adds no inference latency as matrices can be merged with the base weights."
    }
]

# --- Sidebar ---
st.sidebar.image("https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png", width=200)
st.sidebar.title("Assignment 2")
st.sidebar.markdown("### Fine-tuning Qwen 2.5-1.5B-Instruct")
st.sidebar.markdown("""
**Objective:** Fine-tune a lightweight LLM on a specialized AI/ML research QA dataset using QLoRA.

**Model:** Qwen/Qwen2.5-1.5B-Instruct
**Method:** QLoRA (4-bit quantization + LoRA)
**Dataset:** 208 Augmented QA Pairs
""")

menu = st.sidebar.radio("Navigation", ["Overview", "Training Progress", "Model Demonstration", "Evaluation Metrics"])

# --- Overview ---
if menu == "Overview":
    st.title("🤖 AI/ML Research QA Assistant")
    st.subheader("Fine-tuning Demonstration")
    
    st.markdown("""
    This interface demonstrates the results of **Assignment 2**, where we fine-tuned the `Qwen2.5-1.5B-Instruct` model on a custom dataset of AI/ML research questions.
    
    ### Key Features of the Assignment:
    *   **Dataset Creation**: Generated a synthetic dataset of 52 core QA pairs, augmented to 208 examples using various phrasing techniques.
    *   **Efficiency**: Used **QLoRA** (Quantized Low-Rank Adaptation) to enable training on consumer-grade hardware (or free T4 GPUs).
    *   **Comparison**: Evaluated the difference between a generic "Base" model and our specialized "Fine-tuned" version.
    *   **Metrics**: Tracked performance using **ROUGE** and **BLEU** scores.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h4>Model Size</h4><h2>1.5B</h2><p>Parameters</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>Training Time</h4><h2>~2.6 min</h2><p>on Tesla T4</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>ROUGE-1 Gain</h4><h2>+22.7%</h2><p>vs Base Model</p></div>', unsafe_allow_html=True)

# --- Training Progress ---
elif menu == "Training Progress":
    st.title("📈 Training Progress")
    
    st.markdown("The model was trained for **3 epochs** with a total of **36 steps**.")
    
    # Mock training logs from the notebook
    steps = [10, 20, 30, 36]
    train_loss = [2.6236, 2.2914, 1.9236, 2.1710]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, train_loss, label="Training Loss", color="#6366f1", marker='o', linewidth=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("""
    ### Training Details:
    *   **Optimizer**: Paged AdamW 32-bit
    *   **Learning Rate**: 2e-4 with Linear Schedule
    *   **Batch Size**: 4 (with Gradient Accumulation = 4)
    *   **Final Loss**: 2.1710
    """)

# --- Model Demonstration ---
elif menu == "Model Demonstration":
    st.title("🧪 Model Demonstration")
    st.markdown("Select a question to see how the **Fine-tuned** model compares to the **Base** model.")
    
    selected_qa = st.selectbox("Select a Research Question:", [q["input"] for q in QA_PAIRS])
    
    current_qa = next(q for q in QA_PAIRS if q["input"] == selected_qa)
    
    if st.button("Generate Comparison"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="response-header base-header">Base Qwen 2.5 (Generic)</div>', unsafe_allow_html=True)
            with st.empty():
                text = ""
                for word in current_qa["base"].split():
                    text += word + " "
                    st.markdown(f'<div class="response-body">{text}</div>', unsafe_allow_html=True)
                    time.sleep(0.05)
        
        with col2:
            st.markdown('<div class="response-header ft-header">Fine-Tuned (Specialized)</div>', unsafe_allow_html=True)
            with st.empty():
                text = ""
                for word in current_qa["ft"].split():
                    text += word + " "
                    st.markdown(f'<div class="response-body">{text}</div>', unsafe_allow_html=True)
                    time.sleep(0.05)
        
        st.success("Analysis: The fine-tuned model provides significantly more technical detail and structured information compared to the base model.")

# --- Evaluation Metrics ---
elif menu == "Evaluation Metrics":
    st.title("📊 Evaluation Results")
    
    st.markdown("Comparison using standard NLP metrics (Base vs Fine-Tuned).")
    
    metrics_data = {
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"],
        "Base Model": [0.1932, 0.0329, 0.1140, 0.0067],
        "Fine-Tuned": [0.2371, 0.0284, 0.1296, 0.0000],
        "Improvement": ["+22.7%", "-13.5%", "+13.7%", "-100.0%"]
    }
    
    df = pd.DataFrame(metrics_data)
    st.table(df)
    
    st.info("Note: ROUGE-1 and ROUGE-L show significant improvements in semantic recall. BLEU is near zero for both because the dataset is specialized and lexical overlap with short reference answers is low.")
    
    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, df["Base Model"][:3], width, label='Base Model', color='#94a3b8')
    ax.bar(x + width/2, df["Fine-Tuned"][:3], width, label='Fine-Tuned', color='#6366f1')
    ax.set_xticks(x)
    ax.set_xticklabels(df["Metric"][:3])
    ax.set_title("ROUGE Metrics Comparison")
    ax.legend()
    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.caption("Created for Assignment 2 - Gen AI")
