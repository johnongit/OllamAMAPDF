# Context

This repository contains the code for an LLM RetrievalQ&A demonstrator based on Langchain framework.
The goal of this demonstrator is to show how to build RetrievalQ&A system using local LLM.

# Requirements

The demonstrator is based on ollama. To install ollama, please follow the instructions on the [ollama github page](https://github.com/jmorganca/ollama)

# Install Langchain stuff

```bash
pip install langchain

pip install langchain-community
```

# Usage

To run the demonstrator, a PDF file is needed. The PDF file must be in the same directory as the repo.
The first step is to embed the PDF file into the LLM. To do so, run the following command:

```bash
python retrievalqa.py embed pdf_file.pdf
```

This will create a chroma vector folder in the same directory as the PDF file. This folder contains the chroma vectors of the PDF file.

The second step is to run the LLM. To do so, run the following command:

```bash
python retrievalqa.py
```