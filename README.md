# Dynamic Text Tagger

An intelligent, domain-agnostic text tagging system that automatically extracts relevant tags from any text using advanced NLP techniques, linguistic pattern matching, and semantic understanding.

## 🌟 Features

- **Domain-Agnostic**: Works across any domain without pre-training on specific categories
- **Intelligent Phrase Extraction**: Uses spaCy for linguistic analysis to identify meaningful phrases
- **Multi-Factor Scoring**: Combines frequency, specificity, domain relevance, and semantic similarity
- **Redundancy Filtering**: Automatically removes similar/overlapping tags
- **Semantic Understanding**: Uses sentence embeddings to ensure tag relevance
- **Flexible API**: Simple interfaces for different use cases

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install spacy scikit-learn sentence-transformers numpy

# Download spaCy language model
python -m spacy download en_core_web_sm
