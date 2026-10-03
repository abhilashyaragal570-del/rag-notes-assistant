# RAG Notes Assistant (Python + Gemini API)

A command-line app that answers questions from my own notes. It finds the most
relevant paragraphs and asks Gemini to answer using only those paragraphs.

## How it works
1. Load notes.txt and split it into chunks (one per paragraph)
2. Turn every chunk into an embedding (a list of numbers that captures meaning)
3. Embed the question and rank chunks by cosine similarity
4. Send the top 2 chunks to Gemini with the instruction: answer only from these notes
5. If the answer is not in the notes, it says "I don't know based on my notes."

## Tech
Python, Gemini API (embeddings + text generation), python-dotenv

## How to run
1. pip install google-genai python-dotenv
2. Create a .env file next to rag.py with: GEMINI_API_KEY=your_key
3. python rag.py

## What I learned
I built a RAG app that splits a notes file into chunks, embeds them, and finds the
best matches for a question using cosine similarity. I learned to keep my API key in
a .env file so it never reaches GitHub, and to retry when the Gemini API returns
a 503 error.