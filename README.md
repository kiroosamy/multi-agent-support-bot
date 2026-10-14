# 🤖 Multi-Agent Customer Support Bot

An end-to-end intelligent e-commerce customer support platform that uses a custom deep learning intent classifier to route customer inquiries to specialized AI agents. Built with PyTorch, LangChain, FAISS, and Streamlit.

---

## 📌 Table of Contents
- [Architecture](#-architecture)
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [1. Train the Classifier](#1-train-the-classifier-optional)
  - [2. Run the Streamlit App](#2-run-the-streamlit-app)
- [Model Evaluation](#-model-evaluation)
- [Tech Stack](#-tech-stack)

---

## 🏗️ Architecture

```text
               +-----------------------+
               |  User Input (Chat)    |
               +-----------+-----------+
                           |
                           v
         +----------------------------------+
         | Custom PyTorch BiLSTM Classifier |
         | (Predicts Intent & Confidence)   |
         +-----------------+----------------+
                           |
            +--------------+--------------+
            | (High Conf)                 | (Low Conf / < 0.60)
            v                             v
   +-----------------+            +---------------+
   | Router Agent    |            | General Agent |
   +--------+--------+            +---------------+
            |
    +-------+------------------+------------------+
    |                          |                  |
    v                          v                  v
+---------------+      +---------------+   +---------------+
| Billing Agent |      | Shipping Agent|   | Returns Agent |
| (Invoice Tool)|      | (Tracker Tool)|   |  (FAISS RAG)  |
+---------------+      +---------------+   +---------------+
