"""
rag_engine.py
Video RAG engine:
  - Chunks transcript into segments
  - TF-IDF retrieval for relevant sections
  - Google Gemini answers questions based on context
"""

import re
import math
import os
from collections import Counter
from typing import List, Tuple
import google.generativeai as genai


def chunk_transcript(transcript: str, chunk_size: int = 800, overlap: int = 100) -> List[dict]:
    """Split transcript into overlapping chunks, preserving timestamp info."""
    lines  = transcript.splitlines()
    chunks = []
    current, current_len = [], 0

    for line in lines:
        current.append(line)
        current_len += len(line)
        if current_len >= chunk_size:
            text = "\n".join(current).strip()
            # Extract first timestamp found in chunk
            ts_match = re.search(r"\[(\d+:\d+)\]", text)
            timestamp = ts_match.group(1) if ts_match else ""
            if text:
                chunks.append({"text": text, "timestamp": timestamp})
            overlap_lines = current[-3:]  # keep last 3 lines for overlap
            current, current_len = overlap_lines, sum(len(l) for l in overlap_lines)

    remainder = "\n".join(current).strip()
    if remainder:
        ts_match = re.search(r"\[(\d+:\d+)\]", remainder)
        timestamp = ts_match.group(1) if ts_match else ""
        chunks.append({"text": remainder, "timestamp": timestamp})

    return chunks


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


class TFIDFRetriever:
    def __init__(self, chunks: List[dict]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        N = len(self.doc_tokens)
        df = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        self.idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
        self.tfidf_vecs = []
        for tokens in self.doc_tokens:
            tf = Counter(tokens); total = max(len(tokens), 1)
            self.tfidf_vecs.append({t: (c / total) * self.idf.get(t, 1) for t, c in tf.items()})

    def _cosine(self, a, b):
        keys = set(a) & set(b)
        if not keys: return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        ma = math.sqrt(sum(v ** 2 for v in a.values()))
        mb = math.sqrt(sum(v ** 2 for v in b.values()))
        return dot / (ma * mb) if ma and mb else 0.0

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        q = Counter(tokenize(query)); total = max(sum(q.values()), 1)
        qv = {t: (c / total) * self.idf.get(t, 1) for t, c in q.items()}
        scores = [self._cosine(qv, v) for v in self.tfidf_vecs]
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top]


class VideoRAGEngine:
    def __init__(self, transcript: str, api_key: str):
        self.transcript = transcript
        self.api_key    = api_key
        self.chunks     = chunk_transcript(transcript)
        self.retriever  = TFIDFRetriever(self.chunks)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def query(self, question: str, top_k: int = 5) -> Tuple[str, List[str]]:
        relevant = self.retriever.retrieve(question, top_k)

        # Always use top chunks even if low score
        context = "\n\n---\n\n".join(
            f"[Segment at {c['timestamp'] or 'unknown time'}]\n{c['text']}"
            for c in relevant
        )

        timestamps = [c["timestamp"] for c in relevant if c.get("timestamp")]

        prompt = f"""You are an expert video analyst. You have been given transcript segments from a video.

Answer the user's question based on the video transcript below.
Be specific, reference what was said or shown, and mention timestamps when relevant.
If the answer spans multiple parts of the video, mention that.

Video transcript segments:
{context}

Full transcript summary (first 500 words):
{self.transcript[:2000]}

Question: {question}

Answer clearly and helpfully based on the video content."""

        response = self.model.generate_content(prompt)
        return response.text, timestamps