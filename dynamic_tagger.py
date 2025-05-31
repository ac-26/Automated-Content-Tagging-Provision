import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import re
from typing import List, Dict, Tuple
from collections import Counter
import spacy

class TextEncoder:
  #initialization function
  def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
      print(f"Initializing TextEncoder with model: {model_name}", flush=True)
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      print("Tokenizer loaded", flush=True)
      self.model = AutoModel.from_pretrained(model_name)
      print("Model loaded", flush=True)
      self.model.eval()
      print("Model Loaded Successfully", flush=True)

  #encodes text
  def encode_text(self, text: str) -> np.ndarray:
    inputs = self.tokenizer(text, return_tensors="pt",truncation=True, padding=True, max_length=512)
    with torch.no_grad():
      outputs = self.model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()


#This was my initial approach when I was brainstorming to build this system, this is not being used currently
#in our system, IT IS ONLY KEPT FOR DEMONSTRATION OF PROJECT EVOLUTION.
'''
class TagVocabulary:
  #initialization function
  def __init__(self):
    self.tags = [
            # Content Creation
            "Content Writing", "Copywriting", "Blog Writing", "Article Writing",
            "Creative Writing", "Technical Writing", "Content Strategy",

            # Marketing
            "Social Media Marketing", "Digital Marketing", "Email Marketing",
            "Marketing Strategy", "Brand Marketing", "Influencer Marketing",

            # Social Media
            "Social Media", "Facebook Marketing", "Instagram Marketing",
            "Twitter Marketing", "LinkedIn Marketing", "TikTok Marketing",

            # Analytics & Testing
            "A/B Testing", "Analytics", "Performance Tracking", "Data Analysis",
            "Audience Research", "Market Research",

            # Advertising
            "Online Advertising", "Social Media Ads", "Google Ads",
            "Facebook Ads", "Digital Advertising",

            # Skills & Techniques
            "Communication Skills", "Writing Skills", "Creative Skills",
            "Marketing Skills", "Design Skills",

            # Strategy & Planning
            "Content Planning", "Marketing Planning", "Campaign Strategy",
            "Audience Targeting", "Customer Engagement"
        ]

    print(f"Tag vocabulary initialized with {len(self.tags)} tags")

    #this will return list of tags in our vocabulary
    def get_tags(self) -> List[str]:
        return self.tags.copy()

    #this will add a new tag in our vocabulary
    def add_tag(self, new_tag: str):
        if new_tag not in self.tags:
            self.tags.append(new_tag)
            print(f"Added new tag: {new_tag}")
        else:
            print(f"Tag '{new_tag}' already exists")

class BasicTagger:
  #initializer function
    def __init__(self):
      self.encoder = TextEncoder()
      self.vocabulary = TagVocabulary()

      self.tag_embeddings = self._encode_all_tags()

    #function to encode all tags before hand
    def _encode_all_tags(self) -> Dict[str, np.ndarray]:
        tag_embeddings = {}
        for tag in self.vocabulary.tags:
            embedding = self.encoder.encode_text(tag)
            tag_embeddings[tag] = embedding

        return tag_embeddings

    #finds tags from our vocaublary that are applicable according to our text input
    def find_matching_tags(self, input_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        # Encode the input text
        input_embedding = self.encoder.encode_text(input_text)

        similarities = []

        for tag_name, tag_embedding in self.tag_embeddings.items():
            # Calculate cosine similarity
            similarity = cosine_similarity(
                input_embedding.reshape(1, -1),
                tag_embedding.reshape(1, -1)
            )[0][0]

            similarities.append((tag_name, float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

# Test our basic tagger
def test_basic_tagger():
    tagger = BasicTagger()

    test_text = """
    Creating social media posts is a great way to hone your content writing skills.
    Since posts are typically very short, snappy, and quick, you can easily try out
    different styles of writing and see what people respond to. It's easy to change
    direction and adapt if you need to tweak your writing style since social media
    posts are typically fluid and changeable by nature. You can also practice A/B
    testing with your social media ads—try writing two different posts and sending
    it to similar demographics and see which one performs better.
    """

    print("Input text:")
    print(test_text)

    # Find matching tags
    matching_tags = tagger.find_matching_tags(test_text, top_k=15)

    print("Top matching tags:")
    for i, (tag, score) in enumerate(matching_tags, 1):
        print(f"{i:2d}. {tag:<25} (Score: {score:.3f})")

if __name__ == "__main__":
    test_basic_tagger()

'''


# This is the approach that is being followed right now, it is able to fix the problems and limitation that
# we were facing in the above approach

#this extracts key phrases from text using linguistic patterns and statistical methods.
class KeyPhraseExtractor:
    def __init__(self):
        # trying to use spaCy model for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Please install spacy model: python -m spacy download en_core_web_sm")
            raise

        #I have used Parts of Speech(POS) concept for tag extraction
        self.phrase_patterns = [
            #1-word phrases
            ["NOUN"],
            ["PROPN"],

            # 2-word phrases
            ["ADJ", "NOUN"],
            ["ADJ", "PROPN"],
            ["NOUN", "NOUN"],
            ["PROPN", "NOUN"],
            ["NOUN", "PROPN"],
            ["VERB", "NOUN"],

            # 3-word phrases
            ["ADJ", "NOUN", "NOUN"],
            ["NOUN", "NOUN", "NOUN"],
            ["PROPN", "PROPN", "PROPN"],
            ["ADJ", "ADJ", "NOUN"],
            ["NOUN", "VERB", "NOUN"],

            # 4-word phrases
            ["ADJ", "NOUN", "NOUN", "NOUN"],
            ["NOUN", "NOUN", "NOUN", "NOUN"],
            ["ADJ", "ADJ", "NOUN", "NOUN"],
            ["NOUN", "NOUN", "VERB", "NOUN"],
            ["ADJ", "NOUN", "VERB", "NOUN"],
            ["NOUN", "NOUN", "NOUN", "VERB"],

            # Verb forms (gerunds)
            ["VERB"],  # Will filter for -ing forms
            ["NOUN", "VERB"],
            ["ADJ", "VERB"],
        ]

        # Common compound terms that should stay together
        self.compound_terms = {
            "machine learning", "deep learning", "natural language processing",
            "neural network", "data science", "artificial intelligence",
            "computer vision", "big data", "real time", "decision making",
            "supply chain", "customer relationship", "human resources",
            "business process", "electronic health", "patient care"
        }

        print("KeyPhraseExtractor initialized successfully")

    def extract_phrases(self, text: str, min_freq: int = 1) -> List[Tuple[str, int]]:
        #this extracts key phrases on the basis of above defined POS system
        doc = self.nlp(text.lower())

        # Store found phrases with their frequencies
        phrase_counter = Counter()

        # look for known compound terms
        text_lower = text.lower()
        for compound in self.compound_terms:
            # count how many times this compound appears
            count = text_lower.count(compound)
            if count > 0:
                phrase_counter[compound] = count

        # Extract phrases based on POS patterns
        for sentence in doc.sents:
            tokens = list(sentence)

            # Try each starting position
            for start_idx in range(len(tokens)):
                # Try each pattern
                for pattern in self.phrase_patterns:
                    end_idx = start_idx + len(pattern)

                    if end_idx <= len(tokens):
                        # Get tokens for this span
                        span_tokens = tokens[start_idx:end_idx]
                        pos_sequence = [token.pos_ for token in span_tokens]

                        # Check if POS tags match the pattern
                        if pos_sequence == pattern:
                            # Build the phrase
                            phrase_tokens = []
                            valid = True

                            for i, token in enumerate(span_tokens):
                                # Skip stopwords only in single-word phrases
                                if len(pattern) == 1 and token.is_stop:
                                    valid = False
                                    break

                                # For verbs, prefer -ing forms (gerunds)
                                if token.pos_ == "VERB" and len(pattern) == 1:
                                    if not token.text.endswith("ing"):
                                        valid = False
                                        break

                                # Skip very short words in single-word phrases
                                if len(pattern) == 1 and len(token.text) < 3:
                                    valid = False
                                    break

                                phrase_tokens.append(token.text)

                            if valid and phrase_tokens:
                                phrase = " ".join(phrase_tokens)
                                # Clean up the phrase
                                phrase = re.sub(r'\s+', ' ', phrase).strip()

                                # Don't add if it's a subset of a compound term we already found
                                is_subset = False
                                for compound in phrase_counter:
                                    if phrase in compound and phrase != compound:
                                        is_subset = True
                                        break

                                if not is_subset and phrase:
                                    phrase_counter[phrase] += 1

        # Filter by minimum frequency and return
        phrases = [(phrase, freq) for phrase, freq in phrase_counter.items()
                   if freq >= min_freq]

        # Sort by frequency (descending) and then by length (longer first)
        phrases.sort(key=lambda x: (x[1], len(x[0].split())), reverse=True)

        return phrases


class PhraseScorer:
    """
    Scores and filters extracted phrases to identify the best tags.
    REFINED: Better handling of 3-4 word phrases
    """

    def __init__(self):
        # Common/generic words that make poor tags
        self.generic_words = {
            'way', 'ways', 'thing', 'things', 'people', 'person', 'time', 'times',
            'place', 'places', 'day', 'days', 'year', 'years', 'good', 'bad',
            'great', 'nice', 'sure', 'certain', 'different', 'same', 'other',
            'new', 'old', 'high', 'low', 'large', 'small', 'long', 'short',
            'easy', 'hard', 'simple', 'complex', 'nature', 'type', 'types',
            'kind', 'kinds', 'lot', 'lots', 'direction', 'need', 'needs'
        }

        # Words that boost phrase importance
        self.domain_indicators = {
            'analysis', 'strategy', 'marketing', 'development', 'management',
            'design', 'research', 'optimization', 'system', 'process', 'method',
            'technique', 'approach', 'framework', 'model', 'algorithm', 'data',
            'content', 'digital', 'social', 'media', 'online', 'software',
            'testing', 'planning', 'writing', 'creative', 'technical',
            'learning', 'training', 'network', 'neural', 'artificial',
            'intelligence', 'language', 'processing', 'natural', 'automated'
        }

        print("PhraseScorer initialized successfully")

    def calculate_phrase_scores(self, phrases: List[Tuple[str, int]],
                                text_length: int) -> List[Tuple[str, float]]:
        """
        Calculate quality scores for each phrase.
        REFINED: Better scoring for longer phrases
        """
        scored_phrases = []

        # Get max frequency for normalization
        max_freq = max([freq for _, freq in phrases]) if phrases else 1

        for phrase, freq in phrases:
            # Initialize scores
            scores = {
                'frequency': 0.0,
                'specificity': 0.0,
                'length': 0.0,
                'domain_relevance': 0.0,
                'completeness': 0.0
            }

            # 1. Frequency score (normalized, with diminishing returns)
            scores['frequency'] = min(freq / max_freq, 1.0) * 0.3

            # 2. Specificity score (penalize generic phrases)
            words = phrase.lower().split()
            generic_count = sum(1 for word in words if word in self.generic_words)
            scores['specificity'] = (1 - generic_count / len(words)) * 0.25

            # 3. UPDATED Length score - better for 3-4 word phrases
            if len(words) == 1:
                scores['length'] = 0.6
            elif len(words) == 2:
                scores['length'] = 0.85
            elif len(words) == 3:
                scores['length'] = 0.95
            elif len(words) == 4:
                scores['length'] = 1.0  # Best score for 4-word phrases
            else:
                scores['length'] = 0.4  # 5+ words usually too specific
            scores['length'] *= 0.2  # Increased weight from 0.15

            # 4. Domain relevance (contains domain-specific terms)
            domain_word_count = sum(1 for word in words
                                    if word in self.domain_indicators)
            scores['domain_relevance'] = min(domain_word_count / len(words), 1.0) * 0.2

            # 5. Completeness score (avoid partial phrases)
            incomplete_markers = {'of', 'to', 'for', 'with', 'and', 'or', 'but', 'the', 'a', 'an'}
            is_complete = (words[0] not in incomplete_markers and
                           words[-1] not in incomplete_markers)
            scores['completeness'] = 1.0 if is_complete else 0.5
            scores['completeness'] *= 0.05  # Reduced weight

            # Calculate total score
            total_score = sum(scores.values())

            # Bonus for known technical phrases
            technical_phrases = {
                'machine learning', 'deep learning', 'natural language processing',
                'neural network', 'artificial intelligence', 'data science',
                'computer vision', 'big data', 'decision making', 'real time',
                'supply chain management', 'customer relationship management',
                'business process automation', 'electronic health record'
            }

            phrase_lower = phrase.lower()
            # Check if current phrase contains any technical phrase
            for tech_phrase in technical_phrases:
                if tech_phrase in phrase_lower:
                    total_score *= 1.15
                    break

            scored_phrases.append((phrase, total_score))

        # Sort by score (descending)
        scored_phrases.sort(key=lambda x: x[1], reverse=True)

        return scored_phrases

    def filter_similar_phrases(self, scored_phrases: List[Tuple[str, float]],
                               similarity_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Remove similar/redundant phrases, keeping the most meaningful variant.
        REFINED: Smarter filtering that preserves important multi-word phrases
        """
        if not scored_phrases:
            return []

        filtered = []

        for phrase, score in scored_phrases:
            phrase_lower = phrase.lower()
            words = set(phrase_lower.split())

            # Check if we should keep this phrase
            should_keep = True
            phrases_to_remove = []

            for i, (kept_phrase, kept_score) in enumerate(filtered):
                kept_lower = kept_phrase.lower()
                kept_words = set(kept_lower.split())

                # Skip if exact same phrase
                if phrase_lower == kept_lower:
                    should_keep = False
                    break

                # Handle subset relationships
                if words.issubset(kept_words) or kept_words.issubset(words):
                    # Determine which phrase is more valuable
                    len_diff = abs(len(words) - len(kept_words))
                    score_diff = abs(score - kept_score)

                    # Prefer longer phrase if:
                    # 1. Score difference is small (< 0.15)
                    # 2. Length difference is significant (>= 1 word)
                    if score_diff < 0.15 and len_diff >= 1:
                        if len(words) > len(kept_words):
                            # Current phrase is longer and similar score - remove shorter
                            phrases_to_remove.append(i)
                        else:
                            # Kept phrase is longer - don't add current
                            should_keep = False
                            break
                    else:
                        # Large score difference - keep higher scoring one
                        if score > kept_score:
                            phrases_to_remove.append(i)
                        else:
                            should_keep = False
                            break

            # Remove marked phrases
            if phrases_to_remove:
                for idx in reversed(phrases_to_remove):
                    filtered.pop(idx)

            if should_keep:
                filtered.append((phrase, score))

        return filtered

class DynamicTagger:
    """
    Complete dynamic tagging system that:
    1. Extracts key phrases from any text
    2. Scores them based on quality metrics
    3. Uses semantic embeddings to ensure relevance
    4. Returns the best tags for any domain
    """

    def __init__(self, encoder_model="sentence-transformers/all-MiniLM-L6-v2"):
        print("Initializing DynamicTagger...", flush=True)
        self.encoder = TextEncoder(encoder_model)
        print("TextEncoder initialized", flush=True)
        self.extractor = KeyPhraseExtractor()
        print("KeyPhraseExtractor initialized", flush=True)
        self.scorer = PhraseScorer()
        print("PhraseScorer initialized", flush=True)
        print("DynamicTagger ready!", flush=True)

    def generate_tags(self, text: str, max_tags: int = 10, min_score: float = 0.6) -> List[Tuple[str, float]]:
        """
        Generate tags dynamically from input text.

        Args:
            text: Input text to generate tags from
            max_tags: Maximum number of tags to return
            min_score: Minimum quality score for tags

        Returns:
            List of (tag, relevance_score) tuples
        """
        # Step 1: Extract key phrases
        phrases = self.extractor.extract_phrases(text)

        if not phrases:
            return []

        # Step 2: Score phrases for quality
        word_count = len(text.split())
        scored_phrases = self.scorer.calculate_phrase_scores(phrases, word_count)

        # Step 3: Filter redundant phrases
        filtered_phrases = self.scorer.filter_similar_phrases(scored_phrases)

        # Step 4: Apply semantic relevance using embeddings
        text_embedding = self.encoder.encode_text(text)

        # Combine quality score with semantic relevance
        final_scores = []
        for phrase, quality_score in filtered_phrases:
            # Get semantic similarity between phrase and full text
            phrase_embedding = self.encoder.encode_text(phrase)

            # Calculate cosine similarity
            semantic_score = cosine_similarity(
                text_embedding.reshape(1, -1),
                phrase_embedding.reshape(1, -1)
            )[0][0]

            # Combine scores (70% quality, 30% semantic)
            combined_score = (quality_score * 0.7) + (semantic_score * 0.3)

            final_scores.append((phrase, combined_score))

        # Sort by combined score
        final_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply quality threshold
        quality_tags = [(tag, score) for tag, score in final_scores if score > min_score]

        # Ensure minimum number of tags
        if len(quality_tags) < 5 and len(final_scores) >= 5:
            quality_tags = final_scores[:5]

        # Return up to max_tags
        return quality_tags[:max_tags]

    def tag_text(self, text: str, max_tags: int = 7) -> List[str]:
        """
        Simple interface that returns just the tag strings.

        Args:
            text: Input text to tag
            max_tags: Maximum number of tags

        Returns:
            List of tag strings
        """
        tag_scores = self.generate_tags(text, max_tags)
        return [tag for tag, _ in tag_scores]


    def tag_text_with_scores(self, text: str, max_tags: int = 7) -> List[Tuple[str, float]]:
        """
        Interface that returns tags with their scores.

        Args:
            text: Input text to tag
            max_tags: Maximum number of tags

        Returns:
            List of (tag, score) tuples
        """
        return self.generate_tags(text, max_tags)

def test_dynamic_tagger():
    tagger = DynamicTagger()

    # Test 1: Original social media text
    print("="*60)
    print("Test 1: Social Media Marketing Text")
    print("="*60)

    test_text1 = """
    Creating social media posts is a great way to hone your content writing skills.
    Since posts are typically very short, snappy, and quick, you can easily try out
    different styles of writing and see what people respond to. It's easy to change
    direction and adapt if you need to tweak your writing style since social media
    posts are typically fluid and changeable by nature. You can also practice A/B
    testing with your social media ads—try writing two different posts and sending
    it to similar demographics and see which one performs better.
    """

    tags1 = tagger.tag_text_with_scores(test_text1)
    for tag, score in tags1:
        print(f"  '{tag}' - Score: {score:.3f}")


    # Test 2: Technical content
    print("\n" + "="*60)
    print("Test 2: Technical/Programming Text")
    print("="*60)

    test_text2 = """
    Machine learning algorithms are transforming how we process bigdata. Python
    libraries like TensorFlow and PyTorch make it easier to build neural networks
    for deep learning applications. Data scientists use these tools for predictive
    analytics and pattern recognition in complex datasets.
    """

    tags2 = tagger.tag_text_with_scores(test_text2)
    for tag, score in tags2:
        print(f"  '{tag}' - Score: {score:.3f}")

    # Test 3: Medical content
    print("\n" + "="*60)
    print("Test 3: Medical/Healthcare Text")
    print("="*60)

    test_text3 = """
    The patient presented with acute respiratory symptoms including persistent cough
    and shortness of breath. Blood tests revealed elevated white blood cell count.
    Treatment protocol included antibiotics and respiratory therapy. Follow-up
    examination showed significant improvement in lung function.
    """

    tags3 = tagger.tag_text_with_scores(test_text3)
    print("\nFinal tags with scores:")
    for tag, score in tags3:
        print(f"  '{tag}' - Score: {score:.3f}")


    # Test 4: my eg test
    print("\n" + "="*60)
    print("Test 4: My Random Example")
    print("="*60)

    test_text4 = """
    i am eager for the particitation at viewfinder
    """

    tags4 = tagger.tag_text_with_scores(test_text4)
    print("\nFinal tags with scores:")
    for tag, score in tags4:
        print(f"  '{tag}' - Score: {score:.3f}")

# Run the test
if __name__ == "__main__":
    test_dynamic_tagger()

