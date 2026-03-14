

# Bio-SLM RAG Assistant

Bio-SLM RAG Assistant is a lightweight AI assistant designed to answer Biology questions using a Small Language Model combined with Retrieval-Augmented Generation (RAG).
The goal of this project is to explore how smaller models can still provide useful and accurate answers when supported by a domain-specific knowledge base.

The assistant retrieves relevant biology information and injects it into the model prompt before generating a response.

A simple web interface is built using Streamlit so users can interact with the system directly.

## Features

Domain-specific AI assistant for Biology knowledge

Retrieval-Augmented Generation (RAG)

Small Language Model optimization

Semantic search using vector embeddings

Interactive web interface

Simple benchmarking for model performance

## Tech Stack

Programming Language
Python

LLM & Inference
Ollama

RAG Framework
LangChain

Vector Database
ChromaDB

Frontend
Streamlit

API Acceleration
Groq API

## Project Structure

```
SLM_Project
│
├── .devcontainer
│   Development container configuration
│
├── data
│   Biology documents used for retrieval
│
├── benchmark.py
│   Script for measuring model inference performance
│
├── bio_info.json
│   Biology knowledge data used in the assistant
│
├── ui.py
│   Streamlit web interface for the chatbot
│
└── requirements.txt
    Python dependencies
```

## How It Works

The assistant follows a simple RAG pipeline:

1. The user asks a biology-related question.
2. The system retrieves relevant information from the knowledge base.
3. The retrieved content is injected into the prompt.
4. The language model generates an answer using that context.

This approach improves answer accuracy compared to using a standalone model.

## Installation

Clone the repository

```
git clone https://github.com/ntcong2346/SLM_Project.git
cd SLM_Project
```

Install dependencies

```
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit interface

```
streamlit run ui.py
```

Then open the local URL provided by Streamlit in your browser.

## Benchmark

You can test inference performance using:

```
python benchmark.py
```

This script measures token generation speed and response latency.

## Demo

Live application

[https://ntcong2346-slm-project-ui-jrmrpm.streamlit.app/](https://ntcong2346-slm-project-ui-jrmrpm.streamlit.app/)

## Future Improvements

Improve retrieval accuracy with better chunking and embeddings

Add more biology documents to expand the knowledge base

Fine-tune small language models for Vietnamese biology content

Optimize the system to run fully on local machines

## Author

Nguyen T. Cong
University of Engineering and Technology – VNU Hanoi

