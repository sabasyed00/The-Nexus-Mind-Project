#  The Nexus-Mind Project  
*A Comparative RAG-Based LLM Healthcare Assistant (LLaMA2 vs Gemma 2B)*

![Python](https://img.shields.io/badge/python-3.10-blue.svg)  
![License](https://img.shields.io/badge/license-MIT-green.svg)  
![LLM](https://img.shields.io/badge/LLM-Llama2%20%7C%20Gemma2-lightgrey)  
![Status](https://img.shields.io/badge/status-Completed-brightgreen)

---

##  Overview

**Nexus-Mind** is a dual-model AI assistant built to explore how Retrieval-Augmented Generation (RAG) frameworks can enhance healthcare services using Large Language Models (LLMs).  
Developed as an MSc research project at the University of Hertfordshire, it evaluates two models:

- `Nexus-Mind 1.0` → Pre-trained quantized **LLaMA2** model for **offline**, low-resource environments
- `Nexus-Mind 2.0` → Instruction-tuned **Gemma 2B** model on **GPU-enabled cloud (Colab)**

Both systems retrieve trusted medical information and generate context-aware responses — empowering patients and professionals with explainable AI.

---

##  Key Features

-  **RAG Pipeline** using FAISS (CPU) and Chroma (GPU)
-  **Quantized LLMs** for fast, local inference (LLaMA2-7b GGML)
-  **Instruction-Tuned Models** for precision (Gemma 2B)
-  Secure, domain-specific medical corpus (Oxford Handbook, Gale Encyclopedia, etc.)
-  **Interactive Chat UI** using Chainlit and Google Colab
-  Performance Evaluation using real-world healthcare scenarios

---

## System Architecture

### Nexus-Mind 1.0 (LLaMA2 + FAISS)


### Nexus-Mind 2.0 (Gemma2 + Chroma)


---

## 🧪 Datasets

The following resources were used to build the medical knowledge base:

-  *Oxford Handbook of Clinical Medicine* (10th ed.)
-  *Gale Encyclopedia of Medicine* (2nd ed.)
-  *Medical Oncology Handbook* (June 2020 ed.)
-  *A Review of Knowledge-Based Systems in Medical Diagnosis*
-  *Cancer and Cure: A Critical Analysis*

---

##  Installation

###  Clone the repo
```bash
git clone https://github.com/<your-username>/The-Nexus-Mind-Project.git
cd The-Nexus-Mind-Project

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
#See /app/ folder for RAG logic and /notebooks/ for Colab prototype.

N.M 1 cpu deployment
cd app
python run_local_chat.py

Nexus-Mind 2.0 – Google Colab Notebook
Open notebooks/NexusMind2_Cloud.ipynb

Upload your own PDFs or use the provided dataset

Run all cells to interact with the instruction-tuned LLM


