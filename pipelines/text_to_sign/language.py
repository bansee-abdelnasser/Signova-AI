import re
import nltk
from nltk.stem import WordNetLemmatizer
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

lemmatizer = WordNetLemmatizer()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# CLEAN LLM OUTPUT
# -----------------------------
def clean_llm_output(text):
    text = text.strip()
    text = text.split("\n")[0]
    text = re.sub(r'[^A-Z0-9\s]', '', text.upper())
    return text.strip()


# -----------------------------
# CLEAN GLOSS (SAFE)
# -----------------------------
def clean_gloss(gloss):
    gloss = gloss.lower()
    gloss = re.sub(r'[^a-z\s]', '', gloss)
    gloss = re.sub(r'\s+', ' ', gloss)
    return gloss.strip()

# -----------------------------
# MAP WORD TO VOCAB
# -----------------------------
def map_word(word, vocab_set):
    word = word.lower()

    if word in vocab_set:
        return {"original": word, "mapped": word, "type": "direct"}

    # fallback → fingerspelling
    return {"original": word, "mapped": None, "type": "fingerspell"}


# -----------------------------
# LEMMATIZE
# -----------------------------
def lemmatize_words(words):
    return [lemmatizer.lemmatize(w.lower()) for w in words]


# -----------------------------
# LLM → GLOSS (FIXED)
# -----------------------------
def llm_text_to_gloss(sentence):

    prompt = f"""
You are an ASL gloss translator.

TASK:
Convert the sentence into ASL Gloss using ONLY valid vocabulary words.

STRICT RULES:
1. Output ONLY space-separated words (NO hyphens, NO merging words)
2. Do NOT create new words from vocabulary
3. Use ONLY words from the vocabulary list
4. Use uppercase only
5. Remove: IS, AM, ARE, TO, THE, A, AN
6. Keep natural ASL structure (time → topic → action when possible)
7. Output ONE single line only

FINGERSPELLING RULE:
- If a word is NOT in the vocabulary list:
  → spell it letter by letter separated by spaces
  → example: "john" → J O H N

INPUT:
{sentence}


OUTPUT:
"""

    chat = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You output ONLY valid ASL gloss using given vocabulary."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return clean_llm_output(chat.choices[0].message.content)

