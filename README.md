# 🎥 YouTube Video Chatbot using RAG

An AI-powered YouTube Video Chatbot that allows users to interact with YouTube videos by asking questions and receiving context-aware answers directly from video transcripts. Built using **Retrieval-Augmented Generation (RAG)** and advanced retrieval techniques.

## 🚀 Features

- **YouTube Transcript Extraction:** Extracts video transcripts using YouTube Transcript API.
- **Semantic Search:** Converts transcript chunks into vector embeddings using Hugging Face.
- **FAISS Vector Database:** Stores and retrieves relevant transcript chunks efficiently.
- **Multi-Query Retrieval:** Generates multiple variations of user queries to improve retrieval accuracy.
- **MMR Search:** Uses Maximal Marginal Relevance to retrieve diverse and relevant information.
- **Redundancy Filtering:** Removes semantically redundant document chunks.
- **Timestamp Attribution:** Associates generated answers with relevant video timestamps.
- **Context-Grounded Answers:** Generates responses strictly from retrieved transcript information.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG pipeline and chain orchestration |
| Groq API | LLM inference |
| Hugging Face | Text embeddings |
| FAISS | Vector storage and similarity search |
| Streamlit | Interactive web interface |
| YouTube Transcript API | Transcript extraction |

## 🧠 RAG Architecture

The application follows these steps:

1. **Video Input:** User provides a YouTube video URL or video ID.
2. **Transcript Extraction:** Retrieves the English transcript along with timestamps.
3. **Text Chunking:** Splits transcripts into smaller chunks while preserving timestamp metadata.
4. **Embedding Generation:** Converts chunks into vector representations using `all-MiniLM-L6-v2`.
5. **Vector Storage:** Stores embeddings in a FAISS vector database.
6. **Query Transformation:** Generates four alternative queries using an LLM.
7. **Document Retrieval:** Retrieves relevant chunks using MMR search.
8. **Context Optimization:** Removes redundant chunks using embedding-based filtering.
9. **Answer Generation:** Generates context-grounded responses with relevant timestamps.

## 📂 Project Structure

```text
youtube-video-chatbot/
│
├── 1_youtube_chatbot.py    # Main application
├── .env                    # API credentials
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd youtube-video-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit langchain langchain-core langchain-community langchain-huggingface langchain-groq langchain-text-splitters youtube-transcript-api faiss-cpu sentence-transformers python-dotenv
```

### 4. Configure API Key

Create a `.env` file in the project directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the Application

```bash
streamlit run 1_youtube_chatbot.py
```

## 📌 Usage

1. Launch the application.
2. Enter a YouTube video URL or video ID.
3. Enter your question about the video.
4. Click the **Explain** button.
5. Receive a context-aware answer with relevant video timestamps.

## ⚠️ Limitations

- Requires an available English transcript for the YouTube video.
- Videos with disabled or unavailable transcripts cannot be processed.
- Answer quality depends on transcript quality and retrieval relevance.
- Requires an active Groq API key for LLM inference.

## 🔮 Future Improvements

- Implement conversational memory for follow-up questions.
- Introduce transcript and embedding caching.
- Support multiple languages.
- Improve retrieval quality through hybrid search.
- Add source-level evaluation for retrieval and answer accuracy.


---
⭐ If you find this project interesting, consider giving it a star!
