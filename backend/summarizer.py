import os
import re
import requests
from typing import List

# ── Stopwords list for pure-python NLP fallback ─────────────────────────────
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

POS_WORDS = {
    "good", "great", "excellent", "positive", "amazing", "wonderful", "benefit",
    "best", "success", "successful", "love", "like", "happy", "effective", "valuable",
    "progress", "gain", "growth", "advantage", "highlight", "impressive", "quality"
}

NEG_WORDS = {
    "bad", "worst", "terrible", "negative", "poor", "fail", "failure", "error",
    "problem", "issue", "harm", "damage", "loss", "decline", "risk", "hazard",
    "adverse", "difficult", "severe", "threat", "concern", "flaw", "crisis"
}


class TextSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._local_model_attempted = False

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment using TextBlob with pure-python lexicon fallback."""
        if not text or not text.strip():
            return "Neutral"
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.08:
                return "Positive"
            elif polarity < -0.08:
                return "Negative"
            return "Neutral"
        except Exception as e:
            print(f"TextBlob fallback notice: {e}")

        # Lexicon fallback
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        pos_score = sum(1 for w in words if w in POS_WORDS)
        neg_score = sum(1 for w in words if w in NEG_WORDS)

        if pos_score > neg_score:
            return "Positive"
        elif neg_score > pos_score:
            return "Negative"
        return "Neutral"

    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract multi-word key phrases using RAKE or term frequency scoring."""
        if not text or not text.strip():
            return []
        
        # Try RAKE first if NLTK is present
        try:
            from rake_nltk import Rake
            rake = Rake()
            rake.extract_keywords_from_text(text)
            phrases = rake.get_ranked_phrases()
            valid_phrases = [
                p.strip().lower() for p in phrases 
                if 3 <= len(p.strip()) <= 40 and not p.strip().isdigit()
            ]
            if valid_phrases:
                return valid_phrases[:num_keywords]
        except Exception as e:
            print(f"RAKE notice: {e}")

        # Pure-python term-frequency fallback
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        freq = {}
        for w in words:
            if w not in ENGLISH_STOPWORDS:
                freq[w] = freq.get(w, 0) + 1
        
        sorted_words = sorted(freq, key=freq.get, reverse=True)
        return sorted_words[:num_keywords]

    def extractive_summarize(self, text: str, sentences_count: int = 5) -> str:
        """
        Extractive summarization using LexRank (if available) or TF-IDF sentence scoring.
        Returns bullet points separated by newlines.
        """
        if not text or not text.strip():
            return ""

        # Try Sumy LexRank first
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lex_rank import LexRankSummarizer

            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            summary_sentences = summarizer(parser.document, sentences_count)
            bullet_points = [f"• {str(sentence).strip()}" for sentence in summary_sentences if str(sentence).strip()]
            if bullet_points:
                return "\n".join(bullet_points)
        except Exception as e:
            print(f"LexRank notice ({e}). Using pure-Python TF-IDF sentence ranker.")

        # Pure-Python TF-IDF & Position sentence scoring algorithm
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
        if not sentences:
            return text

        if len(sentences) <= sentences_count:
            return "\n".join(f"• {s}" for s in sentences)

        # Word frequency map
        word_freq = {}
        for s in sentences:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', s.lower())
            for w in words:
                if w not in ENGLISH_STOPWORDS:
                    word_freq[w] = word_freq.get(w, 0) + 1

        # Sentence scoring
        sentence_scores = []
        for idx, s in enumerate(sentences):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', s.lower())
            score = sum(word_freq.get(w, 0) for w in words if w not in ENGLISH_STOPWORDS)
            if idx == 0:
                score *= 1.25
            sentence_scores.append((score, idx, s))

        top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:sentences_count]
        top_sentences_in_order = sorted(top_sentences, key=lambda x: x[1])

        return "\n".join(f"• {item[2]}" for item[2] in top_sentences_in_order)

    def _hf_api_summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Serverless abstractive summarization using Hugging Face Inference API.
        """
        models_to_try = [
            self.model_name,
            "facebook/bart-large-cnn",
            "philschmid/bart-large-cnn-samsum"
        ]
        
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        words = text.split()
        if len(words) > 800:
            text = " ".join(words[:800])

        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False
            }
        }

        for model_id in models_to_try:
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=12)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0 and "summary_text" in data[0]:
                        return data[0]["summary_text"].strip()
                print(f"HF Model {model_id} returned HTTP {response.status_code}: {response.text[:100]}")
            except Exception as e:
                print(f"HF API request failed for {model_id}: {e}")

        # Fallback to pure-Python TF-IDF extractive summary if HF API is busy or unauthenticated
        return self.extractive_summarize(text, sentences_count=4).replace("• ", "")

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Summarizes text using Abstractive model (local PyTorch if present, otherwise HF Inference API / TF-IDF).
        """
        if not text or not text.strip():
            return ""

        input_length = len(text.split())
        if input_length < min_length:
            return text

        dynamic_max_length = min(max_length, input_length)
        dynamic_min_length = min(min_length, int(input_length * 0.5))

        if not self._local_model_attempted:
            self._local_model_attempted = True
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            except Exception as e:
                print(f"Local model unavailable ({e}). Using Cloud Inference API.")

        if self._tokenizer is not None and self._model is not None:
            try:
                inputs = self._tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
                summary_ids = self._model.generate(
                    inputs["input_ids"],
                    max_length=dynamic_max_length,
                    min_length=dynamic_min_length,
                    num_beams=4,
                    early_stopping=True
                )
                return self._tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            except Exception as e:
                print(f"Local inference error: {e}")

        return self._hf_api_summarize(text, dynamic_max_length, dynamic_min_length)

# Global summarizer instance
summarizer_instance = TextSummarizer()
