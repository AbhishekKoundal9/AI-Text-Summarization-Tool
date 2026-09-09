import os
import re
import requests
import nltk

# Set up writable directory for NLTK data (required for Vercel / serverless environment)
nltk_data_dir = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

try:
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
except Exception as e:
    print(f"Notice: NLTK download fallback: {e}")

from rake_nltk import Rake
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from textblob import TextBlob


class TextSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._local_model_attempted = False
        self.rake = Rake()
        self.extractive_summarizer = LexRankSummarizer()

    def _get_local_model(self):
        if not self._local_model_attempted:
            self._local_model_attempted = True
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                print(f"Loading local summarization model: {self.model_name}...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                print("Local model loaded successfully.")
            except Exception as e:
                print(f"Local model unavailable ({e}). Using Cloud Inference API.")
        return self._tokenizer, self._model

    def analyze_sentiment(self, text: str) -> str:
        if not text:
            return "Neutral"
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                return "Positive"
            elif polarity < -0.1:
                return "Negative"
            else:
                return "Neutral"
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "Neutral"

    def extract_keywords(self, text: str, num_keywords: int = 5) -> list:
        if not text:
            return []
        try:
            self.rake.extract_keywords_from_text(text)
            phrases = self.rake.get_ranked_phrases()
            if phrases:
                return phrases[:num_keywords]
        except Exception as e:
            print(f"RAKE keyword extraction notice: {e}")
        
        # Word frequency fallback
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stopwords = {"this", "that", "with", "from", "they", "have", "been", "will", "would", "could", "should"}
        freq = {}
        for w in words:
            if w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:num_keywords]

    def extractive_summarize(self, text: str, sentences_count: int = 5) -> str:
        """
        Summarizes text using extractive LexRank algorithm.
        Returns bullet points separated by newlines.
        """
        if not text or len(text.strip()) == 0:
            return ""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summary_sentences = self.extractive_summarizer(parser.document, sentences_count)
            bullet_points = [f"• {str(sentence)}" for sentence in summary_sentences]
            summary = "\n".join(bullet_points)
            if summary.strip():
                return summary
        except Exception as e:
            print(f"Extractive summarization notice: {e}")
        
        # Fallback sentence picker
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chosen = sentences[:sentences_count]
        return "\n".join(f"• {s}" for s in chosen if s.strip())

    def _hf_api_summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Serverless abstractive summarization using Hugging Face Inference API.
        """
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {}
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
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
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0 and "summary_text" in data[0]:
                    return data[0]["summary_text"].strip()
            print(f"HF API returned status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"HF API request failed: {e}")
        
        # Graceful fallback to extractive summary if API is unavailable
        return self.extractive_summarize(text, sentences_count=3)

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Summarizes text using Abstractive model (local PyTorch if present, otherwise HF Inference API).
        """
        if not text or len(text.strip()) == 0:
            return ""

        input_length = len(text.split())
        if input_length < min_length:
            return text

        dynamic_max_length = min(max_length, input_length)
        dynamic_min_length = min(min_length, int(input_length * 0.5))

        # Try local transformers model first if available
        tokenizer, model = self._get_local_model()
        if tokenizer is not None and model is not None:
            try:
                inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=dynamic_max_length,
                    min_length=dynamic_min_length,
                    num_beams=4,
                    early_stopping=True
                )
                return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            except Exception as e:
                print(f"Local inference error: {e}")

        # Use Hugging Face Inference API for serverless environments
        return self._hf_api_summarize(text, dynamic_max_length, dynamic_min_length)

# Global summarizer instance
summarizer_instance = TextSummarizer()
