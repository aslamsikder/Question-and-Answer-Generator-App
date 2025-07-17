# ❓ Question-and-Answer Generator App

This application allows users to upload a PDF document and automatically generates:
- 📌 Questions from the content.
- ✅ Corresponding answers using intelligent retrieval and LLM reasoning.

It is useful for:
- Educational content creation (quizzes, tests),
- Knowledge base building,
- AI-powered document understanding.

---

## 🚀 Features

- 📤 Upload any PDF (e.g., textbook, research paper, notes).
- 🤖 Uses LangChain + OpenRouter + Mistral-7B to:
  - Extract questions from text.
  - Generate accurate answers.
- 📄 Saves output to **CSV** formats.
- 🧠 Smart filtering to separate Q&A cleanly.
- 🌐 Clean web interface (FastAPI + Jinja2 templates).

---

## ⚙️ How to Run

### 1. Create Environment
```bash
conda create -n QAgenerator python=3.11 -y
conda activate QAgenerator
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Huggingface CLI Login using Huggingface token
Login using your Hugging Face token:
```
huggingface-cli login
```
When prompted, paste your Hugging Face access token.
You can get your token from:
https://huggingface.co/settings/tokens

After logging in, the CLI will store your token securely, and you'll be able to use private or gated models like mistralai/Mistral-7B-Instruct.

**Another approach**
To authenticate Hugging Face access through a .env file, you can securely store your Hugging Face token and load it in your Python script using the python-dotenv package.
Create a .env file in your project root with this line:

```bash
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```
In your Python code, load the token and login:
```bash
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load variables from .env
load_dotenv()

# Get token from environment
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Optional: Authenticate using the token
api = HfApi()
api.set_access_token(hf_token)

```
Now your code or any transformers/huggingface_hub functions will authenticate using this token.

### 4. Set API Key (Keep your all API in .env file) - best approach
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
```

### 5. Run the App
```bash
pyhton app.py
```
Then go to: [http://localhost:8000](http://localhost:8000)

---

## 📂 Project Structure

```
Question-and-Answer-Generator-App/
├── app.py                  # FastAPI app logic
└── .env                    # All API Key is stored here
├── research/
│   └── experiment.ipynb    # First I explored the full project here before writing moduler structure code
├── src/
│   ├── __init__.pt.py         
│   ├── helper.py           # PDF reading, question-answer generation logic
│   ├── prompt.py           # All prompt is written here
├── templates/
│   └── index.html          # UI for file upload and display
├── static/
│   └── docs/               # Uploaded files
│   └── output/             # Generated QA results (CSV/Excel/JSON/PDF)
└── README.md
└── requirements.txt        # All required library is saved here
```

---

## 🧰 Tech Stack

- `FastAPI`
- `LangChain`
- `transformers` 
- `mistralai/Mistral-7B-Instruct-v0.1` via Huggingface token for tokenizing
- `mistralai/Mistral-7B-Instruct` via OpenRouter for llmpipeline
- `FAISS` for document chunking
- `PyMuPDF` for PDF reading
---

## 📌 Example Output

| Question | Answer |
|----------|--------|
| What is the purpose of SDGs? | To eliminate poverty and ensure sustainable development. |

---

## ✍️ Author
Developed by **Aslam Sikder**, July 2025  
Email: [aslamsikder.edu@gmail.com](mailto:aslamsikder.edu@gmail.com)  
LinkedIn: [Aslam Sikder - Linkedin](https://www.linkedin.com/in/aslamsikder)  
Google Scholar: [Aslam Sikder - Google Scholar](https://scholar.google.com/citations?hl=en&user=Ip1qQi8AAAAJ)


---
