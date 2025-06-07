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

### 🤗 Try it Online

**[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/ac2607/content-tagger)**

Test the tagger instantly without any installation!

### Installation

```bash
# Install required packages
pip install spacy scikit-learn sentence-transformers numpy

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from dynamic_tagger_step1 import DynamicTagger

# Initialize the tagger
tagger = DynamicTagger()

# Tag some text
text = """
Creating social media posts is a great way to hone your content writing skills. 
You can practice A/B testing with your social media ads and see which posts 
perform better with different demographics.
"""

# Get tags as simple list
tags = tagger.tag_text(text)
print(tags)
# Output: ['social media', 'content writing', 'a/b testing', 'posts', 'demographics']

# Get tags with confidence scores
tags_with_scores = tagger.tag_text_with_scores(text)
for tag, score in tags_with_scores:
    print(f"'{tag}' - Score: {score:.3f}")
```

## 🏗️ Architecture

The system consists of four main components:

### 1. TextEncoder
- Converts text to semantic embeddings using sentence-transformers
- Enables semantic similarity calculations between text and potential tags

### 2. KeyPhraseExtractor
- Uses spaCy for part-of-speech tagging and linguistic analysis
- Identifies phrases based on meaningful POS patterns:
  - Single nouns and proper nouns
  - Adjective + noun combinations
  - Noun compounds
  - Gerunds (verb forms ending in -ing)
- Handles special patterns like "A/B testing"

### 3. PhraseScorer
- Evaluates phrase quality using multiple factors:
  - **Frequency**: How often the phrase appears
  - **Specificity**: Avoids generic words
  - **Length**: Prefers 2-3 word phrases
  - **Domain Relevance**: Boosts domain-specific terms
  - **Completeness**: Ensures phrases are grammatically complete

### 4. DynamicTagger
- Orchestrates the entire pipeline
- Combines quality scores with semantic relevance
- Filters redundant tags and returns the best results

## 📊 Scoring System

Tags are scored using a weighted combination of factors:

- **Quality Score (70%)**:
  - Frequency: 30%
  - Specificity: 25%
  - Length: 15%
  - Domain Relevance: 20%
  - Completeness: 10%

- **Semantic Relevance (30%)**:
  - Cosine similarity between tag and full text embeddings

## 🛠️ Advanced Usage

### Custom Configuration

```python
# Initialize with custom parameters
tagger = DynamicTagger(encoder_model="sentence-transformers/all-MiniLM-L6-v2")

# Generate tags with custom thresholds
tags = tagger.generate_tags(
    text=your_text,
    max_tags=15,          # Maximum number of tags
    min_score=0.5         # Minimum quality threshold
)
```


## 📁 Project Structure

```
dynamic-text-tagger/
├── dynamic_tagger_step1.py    # Main implementation file
├── README.md                  # This file
└── requirements.txt           # Dependencies
```

## 📋 Requirements

Create a `requirements.txt` file:

```txt
spacy>=3.0
scikit-learn
sentence-transformers
numpy
```

## 🎯 Performance Characteristics

- **Accuracy**: High precision through multi-factor scoring
- **Coverage**: Works across diverse domains without retraining
- **Speed**: Efficient processing using optimized NLP libraries
- **Scalability**: Handles texts from short posts to long documents

## 🔍 How It Works

1. **Text Preprocessing**: Clean and prepare input text
2. **Phrase Extraction**: Identify candidate phrases using POS patterns
3. **Quality Scoring**: Evaluate phrases using linguistic and statistical metrics
4. **Semantic Filtering**: Use embeddings to ensure relevance to source text
5. **Redundancy Removal**: Filter out similar/overlapping tags
6. **Final Selection**: Return top-scored, diverse tags

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Experiment with different scoring weights
- Add new POS patterns for phrase extraction
- Try different embedding models
- Extend to other languages (requires different spaCy models)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🔬 Technical Notes

- The system uses spaCy's `en_core_web_sm` model for English text processing
- Sentence embeddings are generated using the `sentence-transformers` library
- Cosine similarity is used to measure semantic relatedness
- The scoring system is tuned for English text but could be adapted for other languages

## 🎉 Example Outputs

**Input**: "Data scientists use machine learning algorithms for predictive analytics and pattern recognition in big data applications."

**Output**: `['data scientists', 'machine learning', 'predictive analytics', 'pattern recognition', 'big data']`

**Input**: "Social media marketing requires engaging content and A/B testing to optimize performance."

**Output**: `['social media marketing', 'engaging content', 'a/b testing', 'performance', 'optimization']`
