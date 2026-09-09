# My RAG Knowledge Agent 🤖

I built this project to practice how AI can answer questions from your own documents. You upload a PDF or Word file, and then you can chat with it like you're talking to someone who already read everything.

---

## What does it actually do?

Basically, you give it your documents (like a resume, a report, anything). It reads them, breaks them into small pieces, and saves them in a special database. When you ask a question, it searches that database to find the most relevant pieces, then uses an AI to write a proper answer.

There are two main steps:

1. **Indexing** — reading your documents and saving them
2. **Querying** — searching them and getting an answer

---

## Tools and libraries I used

| Tool | What it's for |
|------|--------------|
| Google Gemini | The AI that reads and answers (also makes the embeddings) |
| ChromaDB | A database that stores text as numbers so we can search by meaning |
| LangChain | A library that connects everything together |
| Streamlit | Makes the chat interface you see in the browser |
| Python dotenv | Reads the API key from a `.env` file |

---

## How to set it up

### 1. Clone or download the project

Put all the files in one folder on your computer.

### 2. Install the requirements

Open your terminal in that folder and run:

```bash
pip install -r requirements.txt
```

This installs everything the project needs.

### 3. Add your API key

Copy the `_env` file and rename it to `.env`:

```bash
cp _env .env
```

Then open `.env` and paste your Google API key:

```
GOOGLE_API_KEY=your_key_here
```

You can get a free key from [Google AI Studio](https://aistudio.google.com/).

### 4. Put your documents in the data folder

Create a folder called `data` and drop your PDF, DOCX, or TXT files inside it.

---

## How to run it

### Option A — The full web interface (recommended)

```bash
streamlit run appFace.py
```

This opens a browser window. You can upload documents from the sidebar and chat with them right there.

### Option B — Just index your documents first, then test

```bash
python store_embeddings.py   # reads your files and saves them to the database
python rag_agent.py          # asks one test question and prints the answer
```

### Option C — Command line chat

```bash
python chat_agent.py
```

This lets you have a back-and-forth conversation in the terminal. Type `exit` to quit.

---

## What each file does

```
appFace.py          → the Streamlit web app (main interface)
ingest.py           → loads documents and splits them into chunks
store_embeddings.py → converts chunks to vectors and saves to ChromaDB
query.py            → a simple script to test searching the database
rag_agent.py        → asks one question and gets an answer (no chat history)
chat_agent.py       → full conversation in the terminal (remembers previous messages)
```

---

## How it works inside (simple version)

```
Your document
     ↓
Split into small chunks (500 characters each)
     ↓
Each chunk → converted to a vector (numbers) using Gemini
     ↓
Vectors saved in ChromaDB on your computer
     ↓
You ask a question
     ↓
Question → also converted to a vector
     ↓
ChromaDB finds the 3 most similar chunks
     ↓
Those chunks + your question → sent to Gemini
     ↓
Gemini writes an answer based only on what it found
```

The reason we convert text to numbers (vectors) is so the computer can compare meaning, not just exact words. If you ask "What programming languages does he know?" it will still find a chunk that says "Proficient in Python and JavaScript" even though the words are different.

---

## Things I noticed while building this

- The `chunk_overlap=50` setting is important. It makes the end of one chunk repeat at the start of the next, so sentences don't get cut in a bad place.
- `temperature=0.2` on the LLM makes it give more accurate and less creative answers. Good for Q&A.
- You have to use the same embedding model when saving and when searching. If you change it later, the old database becomes useless and you have to re-index everything.
- The web app lets you filter by a specific document. So if you uploaded 5 files, you can make it only search inside one of them.

---

## Known issues / what I want to improve

- [ ] It doesn't support images or tables inside PDFs very well
- [ ] No login system — anyone who runs it can see all the documents
- [ ] Re-indexing deletes the whole database and starts over (slow for large collections)
- [ ] Would be nice to highlight exactly which sentence the answer came from

---

## What I learned from this project

Before this, I didn't understand how chatbots could answer questions about specific documents. Now I understand that the key idea is **retrieval** — you don't give the AI the whole document every time. You first search for the relevant parts, then give only those to the AI. This makes it faster and more accurate.

The concept is called RAG — Retrieval Augmented Generation. Retrieval (search the database), Augmented (add that context to the prompt), Generation (let the LLM write the answer).