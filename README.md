# Multimodal Assistant With Langflow Integration

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Multimodal-Assistant-with-Langflow-integration?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Multimodal-Assistant-with-Langflow-integration?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Multimodal Assistant With Langflow Integration — a thoughtfully built LLM-powered conversational application for intelligent, context-aware dialogue.

---

**Topics:** `agentic-ai` · `conversational-ai` · `deep-learning` · `generative-ai` · `langflow` · `large-language-models` · `multimodal-ai` · `tool-use` · `vision-language-model` · `cross-modal-grounding`

## Overview

Multimodal Assistant With Langflow Integration is an LLM-powered application that provides a clean, functional interface for interacting with frontier language models. It goes beyond a simple prompt-response wrapper: it maintains conversation memory, supports configurable system prompts for persona and task specialisation, and provides streaming output for a responsive chat experience.

The application supports multiple LLM backends — OpenAI GPT-4o, Google Gemini, Anthropic Claude, and locally-running Ollama models — configurable via environment variables without code changes. This makes it a flexible foundation for a wide range of conversational AI use cases: customer support bots, coding assistants, educational tutors, and domain-specific advisors.

The conversation history is managed with configurable memory: either a simple sliding window buffer (last N messages) or a vector-store-backed semantic memory that retrieves relevant earlier context based on the current query. Both modes are implemented and selectable via the sidebar.

---

## Motivation

Building a production-ready LLM chat application requires more than an API call. Memory, streaming, error handling, rate limit management, and a clean UI each add meaningful complexity. This project implements all of these correctly so that it can serve as a solid, reusable foundation for any conversational AI product built on top of it.

---

## Architecture

```
User message → Streamlit chat UI
        │
  Memory retrieval (buffer or vector)
        │
  LLM API call (streamed)
        │
  Response rendering + history update
```

---

## Features

### Multi-Turn Memory
Conversation history maintained across turns — either as a sliding window buffer or semantic vector retrieval for long-context conversations.

### Streaming Token Output
LLM responses are streamed character by character to the UI, providing low-latency perceived response time.

### System Prompt Customisation
Sidebar text area for runtime system prompt editing — change the AI's persona, task focus, or response format without restarting the app.

### Multi-Model Backend
Switch between OpenAI, Gemini, Claude, and Ollama backends via environment variable or sidebar selector.

### Conversation Export
Download full conversation history as Markdown or JSON.

### Token Usage Display
Real-time per-message and session-total token counters in the sidebar.

### Error Handling and Retry
Graceful handling of API rate limits, timeouts, and model errors with automatic retry and user notification.

### Session Management
Create, name, switch between, and delete conversation sessions without page reload.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **Streamlit** | Chat UI | st.chat_message, st.chat_input, sidebar |
| **OpenAI SDK** | Primary LLM backend | GPT-4o function calling and streaming |
| **google-generativeai** | Gemini backend | Gemini Pro/Flash streaming API |
| **python-dotenv** | Config | Environment variable management |
| **FAISS (optional)** | Vector memory | Semantic conversation history retrieval |
| **LangChain (optional)** | Memory abstraction | ConversationBufferMemory wrappers |

> **Key packages detected in this repo:** `streamlit` · `google-generativeai` · `pillow` · `pandas` · `plotly`

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JS projects)
- `pip` or `npm` package manager
- Relevant API keys (see Configuration section)

### Installation

```bash
git clone https://github.com/Devanik21/Multimodal-Assistant-with-Langflow-integration.git
cd Multimodal-Assistant-with-Langflow-integration
python -m venv venv && source venv/bin/activate
pip install streamlit openai google-generativeai python-dotenv
echo 'OPENAI_API_KEY=sk-...' > .env
streamlit run app.py
```

---

## Usage

```bash
streamlit run app.py

# Use Gemini backend
BACKEND=gemini streamlit run app.py

# Use local Ollama
BACKEND=ollama MODEL=llama3 streamlit run app.py
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `(required for OpenAI)` | OpenAI API key |
| `GOOGLE_API_KEY` | `(required for Gemini)` | Google API key |
| `BACKEND` | `openai` | LLM backend: openai, gemini, ollama |
| `MEMORY_WINDOW` | `10` | Number of messages in sliding window memory |

> Copy `.env.example` to `.env` and populate all required values before running.

---

## Project Structure

```
Multimodal-Assistant-with-Langflow-integration/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] RAG mode: chat over uploaded documents with citation
- [ ] Voice I/O with Whisper and TTS
- [ ] Multi-agent mode with specialist sub-agents
- [ ] Docker deployment with Traefik reverse proxy
- [ ] Persistent storage with SQLite conversation database

---

## Contributing

Contributions, issues, and feature requests are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please follow conventional commit messages and ensure any new code is documented.

---

## Notes

API keys for the configured LLM backend are required. Streaming availability depends on the model and backend. Ollama must be running locally for the Ollama backend to work.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Crafted with curiosity, precision, and a belief that good software is worth building well.*
