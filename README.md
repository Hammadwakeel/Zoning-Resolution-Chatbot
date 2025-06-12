Here is a well-structured `README.md` file for your **Zoning Resolution Chatbot** project:

---

# 🏙️ Zoning Resolution Chatbot

A powerful chatbot powered by **LLaMA 3.1 70B** and built with **LangGraph**, **LangChain**, and **Streamlit**, designed to answer user queries about **zoning regulations, land use, and planning**.

---

## 🚀 Features

* ✅ LLM-Powered Natural Language Understanding (LLAMA3.1 70B via Groq)
* ✅ HuggingFace-based embedding for semantic search
* ✅ Smart question routing and rewriting for irrelevant queries
* ✅ Document retrieval via Qdrant vector store
* ✅ Streamlit-based interactive chat interface
* ✅ LangGraph-powered stateful reasoning workflow

---

## 🧠 How It Works

1. **User inputs a question** via chat.
2. **Classifier** checks if the question is related to zoning.
3. If off-topic → Returns "I can't respond to that!".
4. If on-topic → Retrieves documents from a vector store (Qdrant).
5. **Document relevance is graded**.
6. If documents are relevant → Generates answer.
7. If not → Rewrites the query and tries again.

---

## 🛠️ Tech Stack

| Tool          | Role                                                    |
| ------------- | ------------------------------------------------------- |
| `LangGraph`   | Multi-step reasoning workflow                           |
| `LangChain`   | LLM integration, prompt templating, vector store access |
| `ChatGroq`    | Fast and efficient inference for LLaMA 3.1 70B          |
| `Qdrant`      | Vector similarity search for document retrieval         |
| `Streamlit`   | Web interface for chat                                  |
| `HuggingFace` | Sentence embeddings for vector store indexing           |

---

## 📁 Folder Structure

```
.
├── app.py              # Main Streamlit app
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── (vector data & Qdrant hosted externally)
```

---

## 🔐 API Keys & Access

> **Note:** Keep your keys secure. You can store them using environment variables instead of hardcoding.

* **Groq API Key**: `gsk_...`
* **Qdrant API Key**: `AUZL...`
* **HuggingFace Model**: `dunzhang/stella_en_1.5B_v5`

---

## 💻 Running the App Locally

1. **Clone the repo**:

   ```bash
   git clone https://github.com/yourusername/zoning-resolution-chatbot.git
   cd zoning-resolution-chatbot
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

---

## 🧪 Example Queries

* *What are the zoning laws for commercial buildings in Brooklyn?*
* *How do I apply for a zoning variance in NYC?*
* *What is the zoning classification for my property?*

---


