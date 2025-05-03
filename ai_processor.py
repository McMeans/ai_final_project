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
        print("    Processing text with spaCy...")
        # Process text with spaCy
        doc = self.nlp(segment_text)
        
        print("    Extracting named entities...")
        # Extract named entities and filter out duplicates and numerical values
        entities = []
        seen_entities = set()
        for ent in doc.ents:
            # Skip if the entity is purely numerical or contains numpy float values
            if (str(ent.text).replace('.', '').replace(',', '').isdigit() or
                str(ent.text).startswith('np.float') or
                str(ent.text).startswith('float')):
                continue
                
            if (ent.text.lower(), ent.label_) not in seen_entities:
                entities.append((ent.text, ent.label_))
                seen_entities.add((ent.text.lower(), ent.label_))
        
        print("    Creating simple summary...")
        # Create a simple extractive summary
        summary = ""
        if len(segment_text.split()) > 30:
            # Split into sentences
            sentences = [sent.text for sent in doc.sents]
            # Take the first few sentences as summary
            summary = " ".join(sentences[:3])
            
        print("    Analyzing sentiment...")
        # Split text into chunks for sentiment analysis
        sentences = [sent.text for sent in doc.sents]
        chunk_size = 1  # Single sentence per chunk
        sentiment_scores = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i]  # Just take one sentence
            try:
                chunk_sentiment = self.sentiment(chunk)[0]
                sentiment_scores.append(chunk_sentiment['score'])
            except Exception as e:
                print(f"    Warning: Sentiment analysis failed for chunk: {str(e)}")
                continue
        
        # Calculate average sentiment
        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            sentiment = {
                'label': 'positive' if avg_score > 0 else 'negative',
                'score': avg_score
            }
        else:
            sentiment = {'label': 'neutral', 'score': 0.0}
        
        print("    Extracting key phrases...")
        # Extract key phrases using TF-IDF and POS tagging
        words = nltk.word_tokenize(segment_text)
        pos_tags = nltk.pos_tag(words)
        
        # Filter for meaningful phrases (nouns, verbs, adjectives) and exclude numerical values
        meaningful_words = [
            word.lower() for word, tag in pos_tags 
            if (tag.startswith(('NN', 'VB', 'JJ')) and 
                len(word) > 2 and 
                not (str(word).replace('.', '').replace(',', '').isdigit()) and  # Filter out pure numbers
                not str(word).startswith('np.float') and  # Filter out numpy float values
                not str(word).startswith('float') and  # Filter out float values
                not (len(word) > 1 and all(c.isdigit() for c in word)))  # Filter out multi-digit numbers
        ]
        
        if meaningful_words:
            print("    Creating TF-IDF matrix...")
            # Create TF-IDF matrix from meaningful words
            text = ' '.join(meaningful_words)
            tfidf_matrix = self.tfidf.fit_transform([text])
            feature_names = self.tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top phrases with their scores, ensuring they don't contain pure numbers
            key_phrases = [(feature_names[i], float(scores[i])) 
                          for i in scores.argsort()[-5:][::-1]
                          if scores[i] > 0 and  # Only include non-zero scores
                          not (len(feature_names[i]) > 1 and all(c.isdigit() for c in feature_names[i]))]  # Exclude pure numbers
        else:
            key_phrases = []
            
        print("    Analysis complete!")
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
        """Generate suggested commentary using video analysis insights."""
        print("  Analyzing segment text...")
        # Analyze the segment
        analysis = self.analyze_segment(segment_text)
        
        print("  Creating structured commentary...")
        # Create a structured commentary
        commentary_parts = []
        
        # Add visual analysis commentary
        if 'visual_analysis' in segment_text:
            visual_parts = []
            if 'color_palette' in segment_text:
                visual_parts.append("The scene features a rich color palette")
            if 'brightness_levels' in segment_text:
                visual_parts.append("with dynamic lighting")
            if 'scene_changes' in segment_text:
                visual_parts.append("and smooth transitions between shots")
            if visual_parts:
                commentary_parts.append(" ".join(visual_parts) + ".")
        
        # Add audio analysis commentary
        if 'audio_analysis' in segment_text:
            audio_parts = []
            if 'volume_levels' in segment_text:
                audio_parts.append("The audio intensity varies throughout")
            if 'pitch_features' in segment_text:
                audio_parts.append("with a diverse range of tones")
            if audio_parts:
                commentary_parts.append(" ".join(audio_parts) + ".")
        
        # Add speech analysis commentary
        if 'speech_analysis' in segment_text:
            speech_parts = []
            if 'speech_segments' in segment_text:
                speech_parts.append("The dialogue carries emotional weight")
            if speech_parts:
                commentary_parts.append(" ".join(speech_parts) + ".")
        
        # Add key entities and phrases, filtering out numerical values
        if analysis['entities']:
            # Filter out entities that are purely numerical or numpy float values
            entities = [
                f"{ent[0]}" for ent in analysis['entities'] 
                if not str(ent[0]).replace('.', '').replace(',', '').isdigit()  # Filter out pure numbers
                and not str(ent[0]).startswith('np.float')  # Filter out numpy float values
                and not str(ent[0]).startswith('float')  # Filter out float values
                and not any(c.isdigit() for c in str(ent[0]))  # Filter out any numbers
            ]
            if entities:
                commentary_parts.append(f"Key elements include {', '.join(entities)}.")
        
        # If no specific insights were found, create a general commentary
        if not commentary_parts:
            commentary_parts.append("This segment presents a compelling visual and auditory experience.")
            
        print("  Combining commentary parts...")
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