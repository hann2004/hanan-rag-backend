
Copy

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import re
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
 
# --- WORD MATCHING (THE FIX) ---
# The old code used `word in q` which is substring matching.
# "hi" would match inside "think", "hire", "this", etc.
# This function checks for whole words only using regex word boundaries.
def matches_any(text: str, phrases: list[str]) -> bool:
    for phrase in phrases:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text):
            return True
    return False
 
# --- LLM ---
def ask_llm(context: str, question: str, history: list):
    system_prompt = """
You are an AI assistant for Hanan Nasir, a Machine Learning and Backend Developer.
 
Rules:
 - Always answer in a natural, conversational, and honest tone (not like ChatGPT or markdown notes).
 - Always refer to Hanan in the third person (use "she", "her", or "Hanan"), never "I" or "my".
 - Never mention or invent projects that Hanan did not build. Only discuss real projects: Credit Risk Model, Fraud Detection, RAG Chatbot, Ethiopia Financial Inclusion, Empower Library API, Med Tracker, and KAIM projects.
 - When asked about projects, use a human style, but always in third person.
 - When asked about communication, explain that Hanan is clear and practical, focuses on making technical ideas understandable to non-technical users, and gives the credit risk dashboard as an example.
 - When asked for contact, provide: Telegram @Nabii24, GitHub github.com/hann2004, LinkedIn linkedin.com/in/hanan-nasir.
 - For greetings, use: Hi! I'm an AI assistant for Hanan Nasir, a Machine Learning and Backend Developer. Feel free to ask about her projects, skills, or experience.
 - For ending, use: Nice talking to you. See you around 👋
 - Never use markdown formatting, stars, or bullet points. Always sound like a real assistant, not a bot.
"""
    messages = [{"role": "system", "content": system_prompt + "\nContext:\n" + context}]
    for h in history[-6:]:  # slightly increased from 4 to 6 for better memory
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
    q = req.question.lower().strip()
 
    # --- HIRING / FIT QUESTIONS ---
    # Now uses phrase matching so it won't accidentally trigger on unrelated words
    hiring_phrases = [
        "should i hire", "would you hire", "is she a good fit",
        "is she suitable", "would she be a good hire", "should we hire",
        "is she right for", "is she a good candidate", "is she qualified",
        "would you recommend her", "is she worth hiring"
    ]
    if matches_any(q, hiring_phrases):
        return {
            "answer": (
                "If you need someone who can actually build things and not just talk about ML, yes. "
                "She has shipped real projects — a credit risk model with SHAP explainability deployed as a FastAPI, "
                "a fraud detection system with class imbalance handling, and this chatbot you're talking to right now. "
                "She has both backend and ML skills, which is a rare combo at her level. "
                "Want me to walk you through any of her projects in more detail?"
            ),
            "type": "hiring"
        }
 
    # --- NAME ---
    name_phrases = ["your name", "who are you", "what is her name", "full name"]
    if matches_any(q, name_phrases):
        return {"answer": "Her name is Hanan Nasir. I'm just her AI assistant.", "type": "info"}
 
    # --- CONTACT ---
    contact_phrases = ["contact", "email", "telegram", "linkedin", "github", "reach her", "reach out"]
    if matches_any(q, contact_phrases):
        return {
            "answer": (
                "You can reach Hanan through these channels:\n\n"
                "Telegram: @Nabii24\n"
                "GitHub: github.com/hann2004\n"
                "LinkedIn: linkedin.com/in/hanan-nasir\n\n"
                "She's open to internships and junior roles in machine learning and backend development."
            ),
            "type": "contact",
            "contact": {
                "telegram": "@Nabii24",
                "github": "https://github.com/hann2004",
                "linkedin": "https://www.linkedin.com/in/hanan-nasir/"
            }
        }
 
    # --- CV ---
    cv_phrases = ["cv", "resume", "download cv", "her cv"]
    if matches_any(q, cv_phrases):
        return {
            "answer": "You can download Hanan's CV here. If you'd like, I can also highlight key skills or projects from it.",
            "type": "cv",
            "cv_url": "https://drive.google.com/uc?export=download&id=1pKFEVc9FSYynyhRKdJa6060ABKLWBkRw"
        }
 
    # --- GREETING ---
    # Only match these as whole words now — "hi" won't match "think" or "hire"
    greeting_phrases = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "salam", "salaam"]
    if matches_any(q, greeting_phrases):
        return {
            "answer": "Hi! I'm an AI assistant for Hanan Nasir, a Machine Learning and Backend Developer. Feel free to ask me about her projects, skills, or experience.",
            "type": "greeting"
        }
 
    # --- ENDING ---
    ending_phrases = ["thank you", "thanks", "bye", "goodbye", "see you", "take care"]
    if matches_any(q, ending_phrases):
        return {"answer": "Nice talking to you. See you around 👋", "type": "ending"}
 
    # --- EVERYTHING ELSE → LLM ---
    answer = ask_llm(context, req.question, req.history)
    return {"answer": answer, "context": [context]}
 
@app.get("/")
def root():
    return {"msg": "RAG backend is running 🚀"}
