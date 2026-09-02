# 🤖 AI Text Summarization Tool

An AI-powered text summarization application built with Python, FastAPI, Natural Language Processing (NLP), and a web-based frontend. The application provides both abstractive and extractive summarization along with keyword extraction and sentiment analysis.

## 🚀 Features

* ✨ Abstractive text summarization using DistilBART
* 📌 Extractive summarization using LexRank
* 🔑 Automatic keyword extraction using RAKE
* 😊 Sentiment analysis using TextBlob
* 📄 PDF and TXT file processing
* 🌐 Text extraction from web URLs
* 📝 Export summaries as Word documents
* ⚡ FastAPI backend
* 🌐 HTML, CSS, and JavaScript frontend
* 🐳 Docker support
* 🔗 REST API integration

## 🧠 How It Works

The application provides multiple ways to process content:

1. Enter text directly into the application.
2. Upload a PDF or TXT file.
3. Provide a web URL to extract webpage content.
4. Select the required summarization method.
5. The backend processes the content using NLP and AI techniques.
6. The application generates a summary, keywords, and sentiment.
7. The results can be exported as a Word document.

## 🤖 Summarization Methods

### Abstractive Summarization

Uses the `sshleifer/distilbart-cnn-12-6` model through Hugging Face Transformers to generate a new concise summary based on the input text.

### Extractive Summarization

Uses the LexRank algorithm to identify and select the most important sentences from the original text.

## 🔑 Keyword Extraction

The application uses RAKE (Rapid Automatic Keyword Extraction) to identify important keywords from the provided text.

## 😊 Sentiment Analysis

TextBlob is used to analyze the sentiment of the input text and provide a sentiment result.

## 📄 Input Support

The application supports:

* Direct text input
* PDF files
* TXT files
* Webpage URLs

## 📝 Export

Processed results can be exported as a `.docx` Word document containing:

* Summary
* Keywords
* Sentiment

## 🏗️ Project Structure

```text
ML/
│
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── summarizer.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
│
└── .gitignore
```

## 🛠️ Technologies Used

* Python
* FastAPI
* Hugging Face Transformers
* DistilBART
* PyTorch
* Sumy
* LexRank
* RAKE-NLTK
* TextBlob
* PyPDF2
* BeautifulSoup
* Requests
* python-docx
* HTML5
* CSS3
* JavaScript
* Docker

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/AbhishekKoundal9/ML.git
cd ML
```

### Navigate to Backend

```bash
cd backend
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python main.py
```

The FastAPI backend runs on:

```text
http://localhost:8000
```

## 🔌 API Endpoints

### `GET /`

Returns a welcome message from the Text Summarization API.

### `POST /summarize`

Processes text and returns:

* Summary
* Keywords
* Sentiment

The endpoint supports both abstractive and extractive summarization.

### `POST /upload`

Accepts PDF and TXT files and extracts their text content.

### `POST /scrape`

Accepts a webpage URL and extracts text from the webpage.

### `POST /export`

Creates and downloads a Word document containing the generated summary, keywords, and sentiment.

## 🔄 Application Workflow

```text
                User
                  │
        ┌─────────┼─────────┐
        │         │         │
     Text      PDF/TXT     URL
        │         │         │
        └─────────┼─────────┘
                  │
                  ▼
           FastAPI Backend
                  │
                  ▼
            Text Processing
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
    DistilBART  LexRank   RAKE
    Summary    Summary  Keywords
        │         │         │
        └─────────┼─────────┘
                  │
                  ▼
           TextBlob Sentiment
                  │
                  ▼
             Final Results
                  │
                  ▼
          Word Document Export
```

## 🐳 Docker

The backend includes a Dockerfile for containerized deployment.

Build the Docker image:

```bash
docker build -t ai-text-summarizer .
```

Run the container:

```bash
docker run -p 8000:8000 ai-text-summarizer
```

## 🎯 Project Objective

The objective of this project is to develop an AI-powered text processing application that combines Natural Language Processing, Deep Learning, and web technologies.

The application demonstrates practical implementation of text summarization, keyword extraction, sentiment analysis, document processing, and web content extraction in a single platform.

## 🔮 Future Improvements

* Add multilingual text summarization
* Support additional document formats
* Add more summarization models
* Improve summarization quality
* Add summary history
* Add user authentication
* Improve UI/UX
* Add automated testing
* Add model evaluation metrics
* Deploy the application to a cloud platform

## 👨‍💻 Author

**Abhishek Koundal**

GitHub: https://github.com/AbhishekKoundal9

## 📄 License

This project is developed for educational and development purposes.
