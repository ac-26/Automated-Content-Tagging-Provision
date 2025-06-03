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

        for sentence in doc.sents:
            tokens = list(sentence)

            for start_idx in range(len(tokens)):
                for pattern in self.phrase_patterns:
                    end_idx = start_idx + len(pattern)
                    if end_idx <= len(tokens):
                        span_tokens = tokens[start_idx:end_idx]
                        pos_sequence = [token.pos_ for token in span_tokens]

                        if pos_sequence == pattern:
                            phrase_tokens = []
                            valid = True

                            for i, token in enumerate(span_tokens):
                                if len(pattern) == 1 and token.is_stop: #important--------1
                                    valid = False
                                    break

                                if token.pos_ == "VERB" and len(pattern) == 1: #important---------2
                                    if not token.text.endswith("ing"):
                                        valid = False
                                        break

                                if len(pattern) == 1 and len(token.text) < 3: #important-----------3
                                    valid = False
                                    break

                                phrase_tokens.append(token.text)

                            if valid and phrase_tokens:
                                phrase = " ".join(phrase_tokens)
                                #removing and cleaning the joined text
                                phrase = re.sub(r'\s+', ' ', phrase).strip()

                                is_subset = False
                                for compound in phrase_counter:
                                    if phrase in compound and phrase != compound:
                                        is_subset = True
                                        break

                                if not is_subset and phrase:
                                    phrase_counter[phrase] += 1

        #takingff only those tokens that match our minimum frequency
        phrases = [(phrase, freq) for phrase, freq in phrase_counter.items()
                   if freq >= min_freq]

        phrases.sort(key=lambda x: (x[1], len(x[0].split())), reverse=True)

        return phrases


#this is used to score adn filter the phrases that we extracted above to identify best tags in our input sentance
class PhraseScorer:
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
        scored_phrases = []

        # Get max frequency for normalization
        max_freq = max([freq for _, freq in phrases]) if phrases else 1

        for phrase, freq in phrases:
            # Initialize scores
            scores = {
                'frequency': 0.0,
                'specificity': 0.0,
                'length': 0.0,
                # 'domain_relevance': 0.0,
                'completeness': 0.0
            }

            # 1. Frequency score -- more occurring words can have more weight for the sentance
            scores['frequency'] = min(freq / max_freq, 1.0) * 0.3

            # 2. Specificity score -- penalize generic phrases
            words = phrase.lower().split()
            generic_count = sum(1 for word in words if word in self.generic_words)
            scores['specificity'] = (1 - generic_count / len(words)) * 0.25

            # 3. Length score - better for 3-4 word phrases
            if len(words) == 1:
                scores['length'] = 0.6
            elif len(words) == 2:
                scores['length'] = 0.85
            elif len(words) == 3:
                scores['length'] = 0.95
            elif len(words) == 4:
                scores['length'] = 1.0
            else:
                scores['length'] = 0.4
            scores['length'] *= 0.2

            # 4. Completeness score -- avoid partial phrases
            incomplete_markers = {'of', 'to', 'for', 'with', 'and', 'or', 'but', 'the', 'a', 'an'}
            is_complete = (words[0] not in incomplete_markers and
                           words[-1] not in incomplete_markers)
            scores['completeness'] = 1.0 if is_complete else 0.5
            scores['completeness'] *= 0.05  # Reduced weight

            # Calculate total score
            total_score = sum(scores.values())

            # # Bonus for known technical phrases
            # technical_phrases = {
            #     'machine learning', 'deep learning', 'natural language processing',
            #     'neural network', 'artificial intelligence', 'data science',
            #     'computer vision', 'big data', 'decision making', 'real time',
            #     'supply chain management', 'customer relationship management',
            #     'business process automation', 'electronic health record'
            # }

            phrase_lower = phrase.lower()

            # # Check if current phrase contains any technical phrase
            # for tech_phrase in technical_phrases:
            #     if tech_phrase in phrase_lower:
            #         total_score *= 1.15
            #         break

            scored_phrases.append((phrase, total_score))

        scored_phrases.sort(key=lambda x: x[1], reverse=True)

        return scored_phrases


    #This removes similar or redundant phrases keeping the ones that are most meaningful
    def filter_similar_phrases(self, scored_phrases: List[Tuple[str, float]],
                               similarity_threshold: float = 0.5) -> List[Tuple[str, float]]:
        if not scored_phrases:
            return []

        filtered = []

        for phrase, score in scored_phrases:
            phrase_lower = phrase.lower()
            words = set(phrase_lower.split())

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
                    len_diff = abs(len(words) - len(kept_words))
                    score_diff = abs(score - kept_score)

                    # prefer longer phrase if:
                    # score difference is small (< 0.15)
                    # length difference is significant (>= 1 word)
                    if score_diff < 0.15 and len_diff >= 1:
                        if len(words) > len(kept_words):
                            # current phrase is longer and similar score - remove shorter
                            phrases_to_remove.append(i)
                        else:
                            #kept phrase is longer - don't add current
                            should_keep = False
                            break
                    else:
                        #large score difference - keep higher scoring one
                        if score > kept_score:
                            phrases_to_remove.append(i)
                        else:
                            should_keep = False
                            break

            if phrases_to_remove:
                for idx in reversed(phrases_to_remove):
                    filtered.pop(idx)

            if should_keep:
                filtered.append((phrase, score))

        return filtered

#Complete dynamic tagging class
class DynamicTagger:
    def __init__(self, encoder_model="sentence-transformers/all-MiniLM-L6-v2"):
        print("Initializing DynamicTagger...", flush=True)
        self.encoder = TextEncoder(encoder_model)
        print("TextEncoder initialized", flush=True)
        self.extractor = KeyPhraseExtractor()
        print("KeyPhraseExtractor initialized", flush=True)
        self.scorer = PhraseScorer()
        print("PhraseScorer initialized", flush=True)
        print("DynamicTagger ready!", flush=True)


    #Generates tags and scores them
    def generate_tags(self, text: str, max_tags: int = 10, min_score: float = 0.6) -> List[Tuple[str, float]]:
        #extract key phrases
        phrases = self.extractor.extract_phrases(text)

        if not phrases:
            return []

        #score phrases for quality
        word_count = len(text.split())
        scored_phrases = self.scorer.calculate_phrase_scores(phrases, word_count)

        #remove redundant phrases
        filtered_phrases = self.scorer.filter_similar_phrases(scored_phrases)

        #apply semantic relevance using embeddings
        text_embedding = self.encoder.encode_text(text)

        #combine quality score with semantic relevance
        final_scores = []
        for phrase, quality_score in filtered_phrases:
            #get semantic similarity between phrase and full text
            phrase_embedding = self.encoder.encode_text(phrase)

            #calculate cosine similarity
            semantic_score = cosine_similarity(
                text_embedding.reshape(1, -1),
                phrase_embedding.reshape(1, -1)
            )[0][0]

            #combine scores (70% quality, 30% semantic)
            combined_score = (quality_score * 0.7) + (semantic_score * 0.3)

            final_scores.append((phrase, combined_score))

        #sort by combined score
        final_scores.sort(key=lambda x: x[1], reverse=True)

        #apply quality threshold
        quality_tags = [(tag, score) for tag, score in final_scores if score > min_score]

        #ensure minimum number of tags
        if len(quality_tags) < 5 and len(final_scores) >= 5:
            quality_tags = final_scores[:5]

        #return up to max_tags
        return quality_tags[:max_tags]

    #this returns only tags
    def tag_text(self, text: str, max_tags: int = 7) -> List[str]:
        tag_scores = self.generate_tags(text, max_tags)
        return [tag for tag, _ in tag_scores]

    #this returns tags with scores
    def tag_text_with_scores(self, text: str, max_tags: int = 7) -> List[Tuple[str, float]]:
        return self.generate_tags(text, max_tags)

#Testing my model
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

