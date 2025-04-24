import spacy
import torch
from transformers import pipeline
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from base_knowledge_graph import BaseKnowledgeGraph

class AIProcessor:
    def __init__(self, knowledge_graph_path: str = None):
        """Initialize the AI processor with necessary models and tools."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('averaged_perceptron_tagger_eng')
            
        # Load spaCy model for NLP tasks
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize specialized pipelines
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
        
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Initialize transformers for various tasks
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.sentiment = pipeline("sentiment-analysis")
        
        # Initialize TF-IDF vectorizer for keyword extraction
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Load or create base knowledge graph
        self.base_knowledge = BaseKnowledgeGraph.load_from_file(knowledge_graph_path) if knowledge_graph_path else BaseKnowledgeGraph()
        
    def analyze_segment(self, segment_text: str) -> Dict:
        """Perform comprehensive AI analysis on a video segment."""
        # Process text with spaCy
        doc = self.nlp(segment_text)
        
        # Extract named entities and filter out duplicates
        entities = []
        seen_entities = set()
        for ent in doc.ents:
            if (ent.text.lower(), ent.label_) not in seen_entities:
                entities.append((ent.text, ent.label_))
                seen_entities.add((ent.text.lower(), ent.label_))
        
        # Generate summary if text is long enough
        summary = ""
        if len(segment_text.split()) > 30:
            # Set max_length based on input length
            max_length = min(len(segment_text.split()) + 10, 130)
            min_length = max(10, len(segment_text.split()) // 3)
            summary = self.summarizer(segment_text, 
                                    max_length=max_length,
                                    min_length=min_length)[0]['summary_text']
            
        # Analyze sentiment
        sentiment = self.sentiment(segment_text)[0]
        
        # Extract key phrases using TF-IDF and POS tagging
        words = nltk.word_tokenize(segment_text)
        pos_tags = nltk.pos_tag(words)
        
        # Filter for meaningful phrases (nouns, verbs, adjectives)
        meaningful_words = [word.lower() for word, tag in pos_tags 
                          if tag.startswith(('NN', 'VB', 'JJ')) and len(word) > 2]
        
        if meaningful_words:
            # Create TF-IDF matrix from meaningful words
            text = ' '.join(meaningful_words)
            tfidf_matrix = self.tfidf.fit_transform([text])
            feature_names = self.tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top phrases with their scores
            key_phrases = [(feature_names[i], float(scores[i])) 
                          for i in scores.argsort()[-5:][::-1]
                          if scores[i] > 0]  # Only include non-zero scores
        else:
            key_phrases = []
            
        return {
            'entities': entities,
            'summary': summary,
            'sentiment': sentiment,
            'key_phrases': key_phrases
        }
        
    def _extract_and_match_concepts(self, text: str) -> List[Dict]:
        """Extract key phrases and match them with concepts in base knowledge."""
        # Extract keywords using TF-IDF
        tfidf_matrix = self.tfidf.fit_transform([text])
        feature_names = self.tfidf.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        keywords = [(feature_names[i], scores[i]) for i in scores.argsort()[-10:][::-1]]
        
        # Match with base knowledge concepts
        matched_concepts = []
        for keyword, score in keywords:
            # Check if keyword exists in base knowledge
            if keyword in self.base_knowledge.graph:
                concept_type = self.base_knowledge.get_concept_type(keyword)
                related = self.base_knowledge.get_related_concepts(keyword)
                matched_concepts.append({
                    'keyword': keyword,
                    'score': float(score),
                    'type': concept_type,
                    'related_concepts': related
                })
                
        return matched_concepts
        
    def _identify_relevant_concepts(self, text: str) -> List[Dict]:
        """Identify relevant concepts from base knowledge graph."""
        # Get unique node types from the graph
        node_types = set()
        for _, data in self.base_knowledge.graph.nodes(data=True):
            if 'type' in data:
                node_types.add(data['type'])
        categories = list(node_types)
        
        if not categories:
            return []
            
        # Use zero-shot classification to identify relevant concept categories
        results = self.zero_shot(text, categories)
        
        relevant_concepts = []
        for label, score in zip(results['labels'], results['scores']):
            if score > 0.3:  # Threshold for relevance
                # Find nodes of this type
                concepts = [
                    node for node, data in self.base_knowledge.graph.nodes(data=True)
                    if data.get('type') == label
                ]
                relevant_concepts.append({
                    'category': label,
                    'confidence': float(score),
                    'concepts': concepts
                })
                
        return relevant_concepts
        
    def generate_knowledge_graph_data(self, text: str, timestamp: Tuple[float, float]) -> Dict:
        """Generate intelligent knowledge graph data from text using base knowledge."""
        doc = self.nlp(text)
        
        # Extract subject-verb-object triples
        triples = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT":
                    subject = ' '.join([w.text for w in token.lefts if w.dep_ in ["nsubj", "nsubjpass"]])
                    obj = ' '.join([w.text for w in token.rights if w.dep_ in ["dobj", "pobj"]])
                    if subject and obj:
                        # Check if subjects or objects match base knowledge concepts
                        if subject in self.base_knowledge.graph or obj in self.base_knowledge.graph:
                            triples.append((subject, token.text, obj))
        
        # Get relevant concepts from base knowledge
        relevant_concepts = self._identify_relevant_concepts(text)
        
        # Extract descriptors that match our base knowledge
        descriptors = []
        for token in doc:
            if token.text in self.base_knowledge.graph:
                concept_type = self.base_knowledge.get_concept_type(token.text)
                if concept_type == 'descriptor':
                    descriptors.append(token.text)
        
        return {
            'triples': triples,
            'relevant_concepts': relevant_concepts,
            'descriptors': descriptors,
            'timestamp': timestamp
        }
        
    def suggest_commentary(self, segment_text: str) -> str:
        """Generate suggested commentary using base knowledge graph insights."""
        # Analyze the segment
        analysis = self.analyze_segment(segment_text)
        
        # Create a structured commentary
        commentary_parts = []
        
        # Add summary if available
        if analysis['summary']:
            commentary_parts.append(analysis['summary'])
            
        # Add insights from base knowledge graph
        if analysis['relevant_concepts']:
            concepts_text = "This segment demonstrates: " + ", ".join(
                f"{concept['category']} ({', '.join(concept['concepts'][:3])})"
                for concept in analysis['relevant_concepts'][:3]
            )
            commentary_parts.append(concepts_text)
            
        # Add sentiment insight
        sentiment_text = f"The segment conveys a {analysis['sentiment']['label']} tone"
        commentary_parts.append(sentiment_text)
        
        # Add entity information
        if analysis['entities']:
            entities_text = "Key elements mentioned: " + ", ".join([f"{ent[0]} ({ent[1]})" for ent in analysis['entities']])
            commentary_parts.append(entities_text)
            
        return " ".join(commentary_parts)
        
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords using base knowledge graph and NLP."""
        # Get keywords from base knowledge matching
        matched_concepts = self._extract_and_match_concepts(text)
        keywords = [concept['keyword'] for concept in matched_concepts]
        
        # Add named entities
        doc = self.nlp(text)
        entities = [ent.text.lower() for ent in doc.ents]
        
        # Combine and deduplicate
        all_keywords = list(set(keywords + entities))
        return [k for k in all_keywords if len(k) > 3]  # Filter out short words 