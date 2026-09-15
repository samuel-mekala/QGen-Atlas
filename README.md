# 🧠 QGen Atlas - Multilingual Query Generation & Response Validation

[![Live Web Application](https://img.shields.io/badge/Live%20Demo-Render-brightgreen?style=for-the-badge&logo=render)](https://github.com/samuel-mekala/QGen-Atlas)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flask-3.0%2B-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![NLP Engine](https://img.shields.io/badge/NLTK-Chunking-orange?style=for-the-badge&logo=nltk)](https://www.nltk.org/)
[![Translation](https://img.shields.io/badge/Translation-Multilingual-purple?style=for-the-badge&logo=google-translate)](https://github.com/samuel-mekala/QGen-Atlas)
[![Validation](https://img.shields.io/badge/Algorithm-Levenshtein%20Distance-red?style=for-the-badge)](https://github.com/samuel-mekala/QGen-Atlas)

An advanced end-to-end Natural Language Processing (NLP) web framework and assessment platform designed for automated query generation (objective & subjective) from source text, multi-language translation, and string-distance response validation.

🔗 **GitHub Repository**: [https://github.com/samuel-mekala/QGen-Atlas](https://github.com/samuel-mekala/QGen-Atlas)

---

## 📌 Project Overview & Objectives

In modern educational, research, and technical domains, manually constructing comprehension tests and validating user answers is time-consuming and restricted across linguistic boundaries. **QGen Atlas** solves these challenges by combining syntactic parsing, part-of-speech (POS) tagging, regexp chunking, and fuzzy string distance metrics into a unified web-based solution:

1. **Objective Query Generation (Fill-in-the-Blank)**: Syntactically parses source text, extracts key noun phrases, and masks target terms with blanks (`________`) while preserving expected answer keys.
2. **Subjective Conceptual Query Generation**: Identifies subject entities across sentences and formulates conceptual questions (*"What is..."*, *"Explain the concept of..."*, *"Describe the significance of..."*) paired with ground-truth contextual reference answers.
3. **Multilingual Translation Module**: Automatically translates generated questions and answer keys into target languages (Spanish, French, German, Hindi, Tamil, Telugu, Chinese, Japanese, Arabic, Russian, Portuguese, Italian).
4. **Levenshtein Distance Response Validation Engine**: Evaluates user-submitted responses against expected reference answers, calculates exact character edit distance, and computes a calibrated similarity percentage score.

---

## 🔬 Core Architecture & Methodology

```
                     ┌───────────────────────────────┐
                     │     Raw Text Input Document   │
                     └───────────────┬───────────────┘
                                     │
                        Sentence & Word Tokenization
                                     │
                     ┌───────────────┴───────────────┐
                     │   NLTK POS Tagging & Chunking │
                     │   Grammar: NP: {<DT>?<JJ>*<NN.*>+}
                     └───────────────┬───────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌─────────────────────────────┐                     ┌─────────────────────────────┐
│    ObjectiveTest Module     │                     │    SubjectiveTest Module    │
│ Masks Noun Phrase targets   │                     │  Concept Extraction &       │
│ with fill-in-the-blank      │                     │  Pattern-based Questioning  │
└──────────────┬──────────────┘                     └──────────────┬──────────────┘
               │                                                   │
               └─────────────────────────┬─────────────────────────┘
                                         │
                         Multilingual Translation Engine
                            (GoogleTranslator Batch)
                                         │
                         Interactive Assessment Session
                                         │
                         Levenshtein Response Validation
                       (Similarity % & Feedback Scoring)
```

---

## 📏 Response Validation Algorithm

The response validation engine computes similarity percentages using the **Levenshtein Distance Algorithm**:

$$\text{Levenshtein Distance } d(a, b) = \text{Minimum edit operations (insertions, deletions, substitutions) to convert } a \rightarrow b$$

$$\text{Similarity Percentage} = \max\left(0.0, \left(1 - \frac{d(a, b)}{\max(\text{len}(a), \text{len}(b))}\right) \times 100\right)$$

### Feedback Classification Matrix

| Similarity Percentage | Qualitative Grade | Status Badge |
| :---: | :---: | :---: |
| **$\ge 85.0\%$** | **Excellent** | `Success` |
| **$65.0\% - 84.9\%$** | **Good** | `Info` |
| **$40.0\% - 64.9\%$** | **Partial Match** | `Warning` |
| **$< 40.0\%$** | **Incorrect** | `Danger` |

---

## 🌐 Multilingual Support (13+ Supported Languages)

QGen Atlas translates both generated queries and expected answer keys across diverse linguistic preferences:

| Language Code | Language Name | Language Code | Language Name |
| :---: | :---: | :---: | :---: |
| `en` | **English** (Default) | `zh-CN` | **Chinese (Simplified)** |
| `es` | **Spanish** | `ja` | **Japanese** |
| `fr` | **French** | `ar` | **Arabic** |
| `de` | **German** | `ru` | **Russian** |
| `hi` | **Hindi** | `pt` | **Portuguese** |
| `ta` | **Tamil** | `it` | **Italian** |
| `te` | **Telugu** | | |

---

## 🚀 Quickstart Guide (Local Execution)

### 1. Clone the Repository
```bash
git clone https://github.com/samuel-mekala/QGen-Atlas.git
cd QGen-Atlas
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch Local Web Application
```bash
python3 app.py
```
Open your web browser and navigate to: **`http://127.0.0.1:5001/`**

---

## 🧪 Verification & Module Testing

To run the automated verification suite covering tokenization, question generation, translation batching, and Levenshtein similarity:

```bash
PYTHONPATH=. .venv/bin/python3 -c "
from objective import ObjectiveTest
from subjective import SubjectiveTest
from validation import evaluate_response

text = 'Natural Language Processing enables computers to understand human languages.'
obj = ObjectiveTest(text, num_questions=1).generate_questions()
sub = SubjectiveTest(text, num_questions=1).generate_questions()
eval_res = evaluate_response('NLP enables computers to understand languages.', text)

print('Objective Question:', obj[0]['question'])
print('Subjective Question:', sub[0]['question'])
print('Validation Score:', eval_res['similarity_percentage'], '%')
"
```

---

## 🌐 Production Cloud Deployment Guide

### Deploying to Render.com (1-Click Setup)

1. Log in to [Render.com](https://render.com/) with your GitHub account.
2. Click **New + ➔ Web Service** and select `samuel-mekala/QGen-Atlas`.
3. Fill in the build settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`
4. Click **Create Web Service**. Render will build and deploy your application automatically.

---

## 📁 Repository File Structure

```
QGen-Atlas/
├── app.py                            # Flask web application & API route handlers
├── objective.py                      # ObjectiveTest class (NLTK POS tagging & chunking fill-in-blanks)
├── subjective.py                     # SubjectiveTest class (Concept extraction & pattern queries)
├── BERT_translate_custom.py          # Multilingual translation engine (GoogleTranslator batch API)
├── validation.py                     # Levenshtein distance string similarity scoring engine
├── requirements.txt                  # Production Python dependencies (Flask, NLTK, deep-translator, Levenshtein)
├── Procfile                          # Production process config for Gunicorn deployment
├── .gitignore                        # Git exclusion rules (.venv, nltk_data, __pycache__)
├── templates/
│   ├── index.html                    # Main dashboard form interface (Bootstrap & Volt theme)
│   ├── quiz.html                     # Interactive assessment test-taking interface
│   └── results.html                  # Response validation scorecard & Levenshtein metrics
└── README.md                         # Comprehensive project documentation
```

---

## 📜 Author & Acknowledgments

- **Author**: Samuel Mekala
- **Institution / Project**: QGen Atlas - Multilingual NLP Framework
- **GitHub Repository**: [https://github.com/samuel-mekala/QGen-Atlas](https://github.com/samuel-mekala/QGen-Atlas)
