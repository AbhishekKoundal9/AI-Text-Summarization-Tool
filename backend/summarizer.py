from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from rake_nltk import Rake
import os

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from textblob import TextBlob

# Download NLTK data required for RAKE and Sumy
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

class TextSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        print(f"Loading summarization model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.rake = Rake()
        self.extractive_summarizer = LexRankSummarizer()
        print("Model loaded successfully.")

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
            return self.rake.get_ranked_phrases()[:num_keywords]
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

    def extractive_summarize(self, text: str, sentences_count: int = 5) -> str:
        """
        Summarizes the text using extractive LexRank algorithm. Good for lists/tables.
        Returns bullet points separated by newlines.
        """
        if not text or len(text.strip()) == 0:
            return ""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summary_sentences = self.extractive_summarizer(parser.document, sentences_count)
            # Combine sentences with bullet points
            bullet_points = [f"• {str(sentence)}" for sentence in summary_sentences]
            return "\n".join(bullet_points)
        except Exception as e:
            print(f"Error during extractive summarization: {e}")
            return f"Error generating summary: {str(e)}"

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """
        Summarizes the given text using Abstractive BART model.
        """
        if not text or len(text.strip()) == 0:
            return ""

        input_length = len(text.split())
        if input_length < min_length:
            return text

        dynamic_max_length = min(max_length, input_length)
        dynamic_min_length = min(min_length, int(input_length * 0.5))

        try:
            inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=dynamic_max_length,
                min_length=dynamic_min_length,
                num_beams=4,
                early_stopping=True
            )
            summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary_text
        except Exception as e:
            print(f"Error during summarization: {e}")
            return f"Error generating summary: {str(e)}"

# Instantiate a global summarizer object to be used by the API
summarizer_instance = TextSummarizer()
