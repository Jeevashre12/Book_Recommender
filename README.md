📚 Semantic Book Recommender

A semantic book recommendation system that suggests books based on the meaning of a user’s query instead of keyword matching.
Built using modern NLP techniques, vector embeddings, and an interactive web interface.

🚀 Overview

This project uses sentence embeddings and vector search to understand user intent and recommend relevant books.
Users can describe what they want to read in natural language and optionally filter results by category.

✨ Features

Semantic search using sentence embeddings
Category-based filtering
Vector similarity search with ChromaDB
Interactive web UI using Gradio
Clean and simple recommendation interface

🛠️ Tech Stack

Python
LangChain
Sentence Transformers (MiniLM)
ChromaDB
Gradio
Pandas & NumPy

⚙️ How It Works

Book descriptions are converted into vector embeddings
Embeddings are stored in ChromaDB
User queries are embedded and compared using vector similarity
Most relevant books are retrieved and displayed in the UI

▶️ Running the Project
pip install -r requirements.txt
python app.py


The application launches as a local Gradio web interface.

📂 Project Structure
├── app.py
├── books_with_emotions.csv
├── chroma_db/
├── img.png
└── README.md

🎯 Purpose

This project demonstrates practical use of:
Semantic search
Vector databases
NLP-based recommendation systems
Rapid ML UI prototyping
