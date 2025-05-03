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
        """Generate detailed, scene-specific commentary based on video analysis and film theory."""
        # Parse insights from segment text
        insights = {
            'visual': {},
            'audio': {},
            'speech': {},
            'characteristics': {}
        }
        
        current_section = None
        subsection = None
        for line in segment_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Handle main sections
            if line.endswith(':') and not line.startswith(' '):
                current_section = line[:-1]
                subsection = None
                insights[current_section] = {}
                continue
            
            # Handle subsections
            if line.endswith(':') and line.startswith('  '):
                subsection = line.strip()[:-1]
                insights[current_section][subsection] = {}
                continue
            
            # Handle key-value pairs
            if ':' in line:
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    if subsection:
                        insights[current_section][subsection][key] = value
                    else:
                        insights[current_section][key] = value

        # Generate scene-specific commentary
        commentary_parts = []
        
        # Visual Analysis with Scene Context
        if 'visual_analysis' in insights:
            visual = insights['visual_analysis']
            
            # Scene Composition and Blocking
            if 'composition_analysis' in visual:
                comp = visual['composition_analysis']
                if 'rule_of_thirds_scores' in comp:
                    scores = [float(s) for s in comp['rule_of_thirds_scores'].split(', ')]
                    if max(scores) > 0.7:
                        commentary_parts.append("The director's use of the rule of thirds creates a dynamic frame where the characters' positions and movements carry significant narrative weight")
                    elif max(scores) < 0.3:
                        commentary_parts.append("The unconventional framing choices reflect the characters' emotional state and the scene's underlying tension")
                
                if 'symmetry_score' in comp:
                    sym_score = float(comp['symmetry_score'])
                    if sym_score > 0.8:
                        commentary_parts.append("The symmetrical composition mirrors the characters' internal conflicts, creating a visual metaphor for their struggle")
                    elif sym_score < 0.3:
                        commentary_parts.append("The deliberately unbalanced composition visually represents the power dynamics at play in the scene")
            
            # Color and Lighting Analysis
            if 'color_palette' in visual:
                colors = visual['color_palette'].split(', ')
                if colors:
                    color_desc = self._describe_colors(colors)
                    color_meaning = self._interpret_color_meaning(colors)
                    commentary_parts.append(f"The {color_desc} color scheme {color_meaning}, reinforcing the scene's emotional core")
            
            # Scene Transitions
            if 'scene_changes' in visual:
                changes = visual['scene_changes'].split(', ')
                if len(changes) > 3:
                    commentary_parts.append("The rapid editing style creates a sense of urgency, mirroring the characters' escalating situation")
                elif len(changes) == 1:
                    commentary_parts.append("The sustained single take allows the audience to fully immerse themselves in the characters' emotional journey")
        
        # Audio and Dialogue Analysis
        if 'audio_analysis' in insights:
            audio = insights['audio_analysis']
            
            # Sound Design
            if 'volume_levels' in audio:
                vol = float(audio['volume_levels'])
                if vol > 0.7:
                    commentary_parts.append("The heightened sound design amplifies the scene's dramatic impact, drawing attention to key emotional moments")
                elif vol < 0.3:
                    commentary_parts.append("The subtle sound design creates an intimate atmosphere, allowing the characters' quiet moments to resonate")
            
            # Music and Rhythm
            if 'pitch_features' in audio and 'tempo' in audio:
                tempo = float(audio['tempo'])
                if 'pitch_movements' in insights.get('characteristics', {}).get('audio_patterns', {}):
                    pitch_data = insights['characteristics']['audio_patterns']['pitch_movements']
                    if 'range' in pitch_data and 'variability' in pitch_data:
                        pitch_range = float(pitch_data['range'].split('=')[1])
                        variability = float(pitch_data['variability'].split('=')[1])
                        
                        if pitch_range > 1000 and tempo > 120:
                            commentary_parts.append("The dynamic musical score underscores the scene's emotional peaks and valleys, enhancing the narrative tension")
                        elif pitch_range < 500 and tempo < 80:
                            commentary_parts.append("The restrained musical accompaniment allows the characters' emotional journey to take center stage")
        
        # Character and Narrative Analysis
        if 'speech_analysis' in insights:
            speech = insights['speech_analysis']
            
            if 'speech_segments' in speech:
                segments = speech['speech_segments'].split('\n')
                segment_count = len([s for s in segments if 'segment_' in s])
                
                if segment_count > 5:
                    commentary_parts.append("The dense dialogue reveals the complex web of relationships and motivations driving the scene forward")
                elif segment_count == 0:
                    commentary_parts.append("The absence of dialogue speaks volumes about the characters' unspoken emotions and the weight of the moment")
                else:
                    commentary_parts.append("The carefully chosen words carry significant emotional weight, revealing the characters' true feelings beneath the surface")
        
        # Emotional Arc Analysis
        if 'emotional_arc' in insights.get('characteristics', {}):
            arc = insights['characteristics']['emotional_arc']
            if 'start_emotion' in arc and 'end_emotion' in arc:
                start = arc['start_emotion'].split('=')[1]
                end = arc['end_emotion'].split('=')[1]
                if start != end:
                    commentary_parts.append(f"The characters' emotional journey from {start} to {end} reveals their growth and the scene's transformative power")
            
            if 'climax_points' in arc:
                climax_points = arc['climax_points'].split(', ')
                if climax_points:
                    commentary_parts.append("The scene's emotional peaks create a powerful narrative rhythm, building to moments of significant character revelation")
        
        # Combine all parts into a coherent commentary
        if commentary_parts:
            # Add a concluding observation that ties everything together
            dramatic_keywords = ['tension', 'conflict', 'intense']
            overall_tone = "dramatic" if any(keyword in part.lower() for keyword in dramatic_keywords for part in commentary_parts) else "nuanced"
            
            # Add a scene-specific conclusion
            if 'speech_analysis' in insights and 'speech_segments' in insights['speech_analysis']:
                segments = insights['speech_analysis']['speech_segments'].split('\n')
                segment_count = len([s for s in segments if 'segment_' in s])
                if segment_count > 5:
                    conclusion = "The scene's rich dialogue and visual storytelling work in harmony to reveal the characters' complex relationships and motivations"
                else:
                    conclusion = "The scene's visual and audio elements combine to create a powerful emotional experience that speaks to the characters' inner lives"
            else:
                conclusion = f"The {overall_tone} direction effectively combines these elements to create a compelling and meaningful sequence"
            
            commentary_parts.append(conclusion)
            return ". ".join(commentary_parts) + "."
        else:
            # Fallback with base knowledge concepts
            concepts = self.base_knowledge.get_related_concepts('visual_elements') + self.base_knowledge.get_related_concepts('audio_elements')
            return f"The scene demonstrates a thoughtful balance of {', '.join(concepts[:3])}, creating a cohesive viewing experience."

    def _interpret_color_meaning(self, colors: List[str]) -> str:
        """Interpret the meaning of color choices based on film theory."""
        warm_count = 0
        cool_count = 0
        bright_count = 0
        dark_count = 0
        
        for color in colors:
            try:
                if isinstance(color, str):
                    if color.startswith('(') and color.endswith(')'):
                        r, g, b = map(int, color.strip('()').split(','))
                    else:
                        continue
                else:
                    r, g, b = color
                
                brightness = (r + g + b) / 3
                if brightness > 200:
                    bright_count += 1
                elif brightness < 50:
                    dark_count += 1
                
                if r > g and r > b:
                    warm_count += 1
                elif b > r and b > g:
                    cool_count += 1
                    
            except (ValueError, TypeError):
                continue
        
        if warm_count > cool_count:
            if bright_count > dark_count:
                return "suggests passion and energy"
            else:
                return "creates a sense of danger or intensity"
        elif cool_count > warm_count:
            if bright_count > dark_count:
                return "evokes calm and clarity"
            else:
                return "establishes a mood of isolation or melancholy"
        else:
            return "maintains a balanced emotional tone"

    def _analyze_emotions(self, emotions: List[str]) -> str:
        """Analyze the emotional content of speech segments."""
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if not emotion_counts:
            return ""
            
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        if dominant_emotion in ['anger', 'fear']:
            return f"The dominant {dominant_emotion} in the dialogue creates tension and conflict"
        elif dominant_emotion in ['joy', 'excitement']:
            return f"The prevailing {dominant_emotion} establishes an uplifting atmosphere"
        elif dominant_emotion in ['sadness', 'melancholy']:
            return f"The underlying {dominant_emotion} adds depth to the character interactions"
        else:
            return f"The {dominant_emotion} in the dialogue shapes the emotional landscape of the scene"

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

    def _describe_colors(self, colors: List[str]) -> str:
        """Convert color values to descriptive text."""
        if not colors:
            return "neutral"
            
        # Convert RGB values to descriptive terms
        color_descriptions = []
        for color in colors:
            try:
                # Handle different color formats
                if isinstance(color, str):
                    if color.startswith('(') and color.endswith(')'):
                        # RGB tuple string
                        r, g, b = map(int, color.strip('()').split(','))
                    else:
                        # Skip if not a valid color format
                        continue
                else:
                    # Assume it's already a tuple
                    r, g, b = color
                
                # Calculate brightness
                brightness = (r + g + b) / 3
                
                # Determine color family
                if r > g and r > b:
                    if brightness > 200:
                        color_descriptions.append("bright red")
                    elif brightness < 50:
                        color_descriptions.append("deep red")
                    else:
                        color_descriptions.append("warm red")
                elif g > r and g > b:
                    if brightness > 200:
                        color_descriptions.append("bright green")
                    elif brightness < 50:
                        color_descriptions.append("deep green")
                    else:
                        color_descriptions.append("natural green")
                elif b > r and b > g:
                    if brightness > 200:
                        color_descriptions.append("bright blue")
                    elif brightness < 50:
                        color_descriptions.append("deep blue")
                    else:
                        color_descriptions.append("cool blue")
                else:
                    if brightness > 200:
                        color_descriptions.append("bright neutral")
                    elif brightness < 50:
                        color_descriptions.append("deep neutral")
                    else:
                        color_descriptions.append("balanced neutral")
                        
            except (ValueError, TypeError):
                # Skip invalid color values
                continue
                
        # Remove duplicates and limit to top 3 colors
        unique_colors = list(dict.fromkeys(color_descriptions))[:3]
        
        if not unique_colors:
            return "neutral"
        elif len(unique_colors) == 1:
            return unique_colors[0]
        elif len(unique_colors) == 2:
            return f"{unique_colors[0]} and {unique_colors[1]}"
        else:
            return f"{unique_colors[0]}, {unique_colors[1]}, and {unique_colors[2]}" 