from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import requests

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data/portfolio_context.txt')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Check your environment variables.")
LLM_MODEL = "llama-3.1-8b-instant"

# --- LOAD DATA ---
def load_portfolio_data():
    with open(DATA_PATH, 'r') as f:
        context = f.read()
    return context

# --- LLM ---
def ask_llm(context: str, question: str, history: list):
    system_prompt = """
You are an AI assistant for Hanan Nasir, a Machine Learning and Backend Developer.

Rules:
 - Always answer in a natural, conversational, and honest tone (not like ChatGPT or markdown notes).
 - Always refer to Hanan in the third person (use \"she\", \"her\", or \"Hanan\"), never \"I\" or \"my\".
 - Never mention or invent projects that Hanan did not build. Only discuss real projects: Credit Risk Model, Fraud Detection, RAG Chatbot, Ethiopia Financial Inclusion, Empower Library API, Med Tracker, and KAIM projects.
 - When asked about projects, use a human style, but always in third person. Example:
     Hanan has worked on several machine learning and backend projects, mostly focused on real-world problems. One of her strongest projects is a credit risk modeling system. It's an end-to-end ML pipeline where she worked on data preprocessing, feature engineering, and model training. She also implemented temporal data splitting to avoid data leakage, which is important in financial systems. What makes it stronger is that she didn’t stop at the model. She added explainability using SHAP, deployed it using FastAPI, and built a Streamlit dashboard so it can actually be used by non-technical users. She has also worked on fraud detection systems and other data science projects during her training, focusing on solving practical problems rather than just theory. She prefers learning by building real projects rather than just studying theory, which is why most of her experience comes from hands-on work. If you want, the assistant can walk you through one project in detail.
 - When asked about communication, use this answer:
     Hanan is clear and practical in her communication. She focuses on explaining technical ideas in a simple and understandable way, especially when working on projects that involve non-technical users. For example, in her credit risk project, she built a dashboard to make the model outputs easier to understand instead of keeping everything technical. She’s also a good listener and asks the right questions to understand problems before solving them. This helps her build more effective and relevant solutions. She enjoys learning continuously and is open to collaborating and sharing ideas with others. You can find her work on GitHub, and she’s also available on LinkedIn and Telegram if you’d like to connect.
 - When asked for CV, say: You can download Hanan’s CV here: 👉 Download CV. If you'd like, the assistant can also highlight key skills or projects from it.
 - When asked for contact, say: You can reach Hanan through: Telegram: @Nabii24, GitHub: github.com/hann2004, LinkedIn: (real link). She’s open to internships and junior roles in machine learning and backend development.
 - For greetings, use: Hi! I’m an AI assistant for Hanan Nasir, a Machine Learning and Backend Developer. Feel free to ask about her projects, skills, or experience.
 - For ending, use: Nice talking to you. See you around 👋
 - Never use markdown formatting, stars, or bullet points for project lists. Always sound like a real assistant, not a bot.
"""
    messages = [{"role": "system", "content": system_prompt + "\nContext:\n" + context}]
    for h in history[-4:]:
        messages.append(h)
    messages.append({"role": "user", "content": question})
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 400
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=data)
    if resp.ok:
        return resp.json()["choices"][0]["message"]["content"]
    return "Something went wrong 😅"

# --- FASTAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
context = load_portfolio_data()

class QueryRequest(BaseModel):
    question: str
    history: List[dict] = []

@app.post("/rag-query")
def rag_query(req: QueryRequest):
    q = req.question.lower()
    # Smart rules for hiring/fit questions
    if any(word in q for word in ["should i hire", "would you hire", "is she a good fit", "is she suitable", "would she be a good hire", "would you recommend", "should we hire", "is she right for", "is she a good candidate", "is she qualified"]):
        return {
            "answer": (
                "As an AI assistant, I can’t make hiring decisions, but I can share more about Hanan’s skills, experience, and work style if you’d like! If you have specific requirements or questions, just let me know."
            ),
            "type": "hiring"
        }
    # Smart rules for name
    if any(word in q for word in ["your name", "who are you", "what is her name", "full name"]):
        return {"answer": "Her name is Hanan Nasir. I’m just her AI assistant.", "type": "info"}
    # Smart rules for contact
    if any(word in q for word in ["contact", "email", "telegram", "linkedin", "github"]):
        return {
            "answer": (
                "You can reach Hanan through these channels:\n\n"
                "Telegram: @Nabii24\n"
                "GitHub: github.com/hann2004\n"
                "LinkedIn: linkedin.com/in/hanan-nasir\n\n"
                "She’s open to internships and junior roles in machine learning and backend development."
            ),
            "type": "contact",
            "contact": {
                "telegram": "@Nabii24",
                "github": "https://github.com/hann2004",
                "linkedin": "https://www.linkedin.com/in/hanan-nasir/"
            }
        }
    # Smart rules for CV
    if "cv" in q or "resume" in q:
        return {
            "answer": (
                "You can download Hanan’s CV here. If you'd like, I can also highlight key skills or projects from it."
            ),
            "type": "cv",
            "cv_url": "https://drive.google.com/uc?export=download&id=1pKFEVc9FSYynyhRKdJa6060ABKLWBkRw"
        }
    # Smart rules for greeting
    if any(word in q for word in ["hello", "hi", "hey", "greetings"]):
        return {"answer": "Hi! I’m an AI assistant for Hanan Nasir, a Machine Learning and Backend Developer. Feel free to ask me about her projects, skills, or experience.", "type": "greeting"}
    # Smart rules for ending
    if any(word in q for word in ["thank you", "thanks", "bye", "goodbye"]):
        return {"answer": "Nice talking to you. See you around 👋", "type": "ending"}
    # Retrieval (just use context)
    answer = ask_llm(context, req.question, req.history)
    return {"answer": answer, "context": [context]}

@app.get("/")
def root():
    return {"msg": "RAG backend is running 🚀"}