# System imports
import os
from typing import List, Dict, Tuple

# Disable GPU/MPS before anything else
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_MPS"] = "0"
os.environ["TORCH_DEVICE"] = "cpu"

# Third-party imports
import numpy as np
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import gc

# Torch and transformers
import torch
torch.set_default_tensor_type('torch.FloatTensor')  # Force CPU tensor type
from transformers import pipeline

# Local imports
from base_knowledge_graph import BaseKnowledgeGraph

class AIProcessor:
    def __init__(self, knowledge_graph_path: str = None):
        """Initialize the AI processor with necessary models and tools."""
        print("Initializing NLP models on CPU...")
        
        try:
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
            print("Loading spaCy model...")
            self.nlp = spacy.load('en_core_web_sm')
            
            # Initialize specialized pipelines
            print("Loading emotion classifier...")
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                use_fast=True,
                framework="pt"
            )
            
            print("Loading zero-shot classifier...")
            self.zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                use_fast=True,
                framework="pt"
            )
            
            # Initialize transformers for various tasks
            print("Loading summarizer...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                use_fast=True,
                framework="pt"
            )
            
            print("Loading sentiment analyzer...")
            self.sentiment = pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                use_fast=True,
                framework="pt"
            )
            
            # Initialize TF-IDF vectorizer for keyword extraction
            self.tfidf = TfidfVectorizer(stop_words='english')
            
            # Load or create base knowledge graph
            print("Loading knowledge graph...")
            self.base_knowledge = BaseKnowledgeGraph.load_from_file(knowledge_graph_path) if knowledge_graph_path else BaseKnowledgeGraph()
            
            # Clear any unused memory
            gc.collect()
            
            print("All NLP models initialized successfully")
            
        except Exception as e:
            print(f"Error during NLP model initialization: {str(e)}")
            raise
        
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
        
    def suggest_commentary(self, analysis_data: Dict) -> str:
        """Generate detailed, scene-specific commentary based on video analysis."""
        try:
            commentary_parts = []
            
            # Analyze Visual Composition
            if 'visual_analysis' in analysis_data:
                visual = analysis_data['visual_analysis']
                
                # Composition Analysis
                if 'composition' in visual and visual['composition']:
                    compositions = visual['composition']
                    if isinstance(compositions, list) and compositions:
                        # Analyze composition trends
                        symmetry_scores = [comp.get('symmetry_score', 0) for comp in compositions]
                        thirds_scores = [comp.get('rule_of_thirds_intensity', 0) for comp in compositions]
                        edge_density = [comp.get('edge_density', 0) for comp in compositions]
                        
                        # Overall composition style
                        avg_symmetry = np.mean(symmetry_scores)
                        avg_thirds = np.mean(thirds_scores)
                        composition_variance = np.std(edge_density)
                        
                        # More specific symmetry observations
                        if avg_symmetry > 0.8:
                            commentary_parts.append("The near-perfect symmetry creates a sense of order and control, suggesting the characters' mastery of their environment")
                        elif avg_symmetry > 0.7:
                            commentary_parts.append("The balanced composition reflects the scene's equilibrium, with each element carefully positioned to maintain visual harmony")
                        elif avg_symmetry < 0.2:
                            commentary_parts.append("The extreme asymmetry destabilizes the frame, mirroring the characters' emotional turmoil")
                        elif avg_symmetry < 0.3:
                            commentary_parts.append("The deliberately off-balance framing creates a sense of unease, hinting at underlying tensions")
                        
                        # More nuanced rule of thirds analysis
                        if avg_thirds > 0.8:
                            commentary_parts.append("The precise application of the rule of thirds creates a dynamic tension between foreground and background elements")
                        elif avg_thirds > 0.6:
                            commentary_parts.append("The strategic placement of key elements along the thirds lines guides the viewer's attention through the scene")
                        
                        # More detailed composition variance analysis
                        if composition_variance > 0.3:
                            commentary_parts.append("The dramatic shifts in visual complexity create a rhythmic pattern that mirrors the scene's emotional arc")
                        elif composition_variance > 0.2:
                            commentary_parts.append("The evolving frame density suggests a deliberate progression in the scene's visual storytelling")
                
                # Enhanced Color Analysis
                if 'color_analysis' in visual and visual['color_analysis']:
                    colors = visual['color_analysis']
                    if isinstance(colors, list) and colors:
                        # Analyze color trends
                        saturation_levels = [c['color_stats'].get('average_saturation', 0) for c in colors]
                        hue_levels = [c['color_stats'].get('average_hue', 0) for c in colors]
                        dominant_colors = [c.get('dominant_colors', []) for c in colors]
                        
                        avg_saturation = np.mean(saturation_levels)
                        hue_variance = np.std(hue_levels)
                        
                        # More specific color observations
                        if avg_saturation > 180:
                            commentary_parts.append("The vibrant, saturated colors create an almost surreal atmosphere, heightening the scene's emotional intensity")
                        elif avg_saturation > 128:
                            commentary_parts.append("The rich color palette enhances the scene's visual impact, drawing attention to key narrative elements")
                        elif avg_saturation < 40:
                            commentary_parts.append("The desaturated tones create a stark, minimalist aesthetic that emphasizes the scene's emotional weight")
                        
                        # Analyze color transitions
                        if hue_variance > 40:
                            commentary_parts.append("The bold color transitions create a visual rhythm that underscores the scene's dramatic progression")
                        elif hue_variance > 30:
                            commentary_parts.append("The subtle shifts in color temperature reflect the evolving emotional landscape of the scene")
                        
                        # Analyze dominant colors
                        if dominant_colors:
                            color_meaning = self._interpret_color_meaning(dominant_colors[0])
                            commentary_parts.append(f"The dominant color scheme {color_meaning}")
                
                # Enhanced Lighting Analysis
                if 'lighting' in visual and visual['lighting']:
                    lighting = visual['lighting']
                    if isinstance(lighting, list) and lighting:
                        # Analyze lighting trends
                        brightness_levels = [light.get('brightness', 0) for light in lighting]
                        contrast_levels = [light.get('contrast', 0) for light in lighting]
                        shadow_ratios = [light.get('histogram_stats', {}).get('shadows', 0) for light in lighting]
                        gradient_stats = [light.get('gradient_stats', {}) for light in lighting]
                        
                        avg_brightness = np.mean(brightness_levels)
                        avg_contrast = np.mean(contrast_levels)
                        shadow_presence = np.mean(shadow_ratios)
                        
                        # More specific lighting observations
                        if avg_brightness > 200:
                            commentary_parts.append("The high-key lighting creates an almost ethereal atmosphere, suggesting a moment of revelation or clarity")
                        elif avg_brightness > 180:
                            commentary_parts.append("The bright, even lighting exposes every detail, creating a sense of transparency and honesty")
                        elif avg_brightness < 50:
                            commentary_parts.append("The deep shadows and low-key lighting create a sense of mystery and psychological depth")
                        
                        # More nuanced contrast analysis
                        if avg_contrast > 70:
                            commentary_parts.append("The stark contrast between light and shadow creates a dramatic chiaroscuro effect, emphasizing the scene's emotional extremes")
                        elif avg_contrast > 50:
                            commentary_parts.append("The strong contrast sculpts the scene with light and shadow, adding depth and dimension to the visual storytelling")
                        
                        # More detailed shadow analysis
                        if shadow_presence > 0.6:
                            commentary_parts.append("The pervasive shadows create a sense of foreboding, suggesting hidden depths and unspoken tensions")
                        elif shadow_presence > 0.4:
                            commentary_parts.append("The strategic use of shadows adds layers of meaning to the scene, creating visual subtext")
            
            # Enhanced Scene Dynamics Analysis
            if 'scene_analysis' in analysis_data:
                scene = analysis_data['scene_analysis']
                
                # Enhanced Pacing Analysis
                if 'pacing' in scene and 'error' not in scene['pacing']:
                    pacing = scene['pacing']
                    visual_pace = pacing.get('visual_pace', {})
                    audio_pace = pacing.get('audio_pace', {})
                    
                    # More specific pacing observations
                    if visual_pace.get('variance', 0) > 0.3:
                        commentary_parts.append("The dynamic visual rhythm creates a sense of urgency and unpredictability, keeping the audience engaged")
                    elif visual_pace.get('variance', 0) > 0.2:
                        commentary_parts.append("The varying pace of visual changes creates a natural ebb and flow that mirrors the scene's emotional journey")
                    elif visual_pace.get('average', 0) > 0.8:
                        commentary_parts.append("The rapid-fire visual changes create a sense of controlled chaos, driving the scene forward with relentless energy")
                    elif visual_pace.get('average', 0) > 0.7:
                        commentary_parts.append("The quick pacing maintains a sense of momentum while allowing key moments to resonate")
                    else:
                        commentary_parts.append("The deliberate pacing gives each moment room to breathe, creating space for emotional resonance")
                
                # Enhanced Emotional Tone Analysis
                if 'emotional_tone' in scene and 'error' not in scene['emotional_tone']:
                    tone = scene['emotional_tone']
                    if 'lighting_mood' in tone:
                        mood = tone['lighting_mood']
                        if mood.get('contrast_level', 0) > 0.7:
                            commentary_parts.append("The extreme contrast in lighting creates a visual metaphor for the scene's emotional extremes")
                        elif mood.get('contrast_level', 0) > 0.5:
                            commentary_parts.append("The interplay of light and shadow creates a rich emotional texture that underscores the scene's complexity")
                
                # Enhanced Key Moments Analysis
                if 'key_moments' in scene:
                    moments = scene['key_moments']
                    if isinstance(moments, list):
                        if len(moments) > 3:
                            commentary_parts.append("The scene builds through a series of carefully orchestrated moments, each contributing to a larger emotional arc")
                        elif len(moments) > 1:
                            commentary_parts.append("The scene's key moments create a narrative rhythm that guides the audience through its emotional journey")
            
            # Combine into final commentary
            if commentary_parts:
                # Select a more varied concluding observation
                if len(commentary_parts) > 1:
                    conclusions = [
                        "These carefully crafted visual elements work in harmony to create a rich cinematic experience",
                        "The scene's technical choices serve its emotional core, creating a powerful viewing experience",
                        "These cinematic techniques combine to tell a story that transcends the visual medium",
                        "The scene's visual language speaks volumes about its underlying themes and emotions"
                    ]
                    conclusion = np.random.choice(conclusions)
                    commentary_parts.append(conclusion)
                commentary = ". ".join(commentary_parts) + "."
            else:
                # More varied fallback commentary
                fallbacks = [
                    "The scene demonstrates a masterful use of visual storytelling techniques",
                    "The cinematography creates a rich visual tapestry that enhances the narrative",
                    "The scene's visual elements work together to create a compelling cinematic experience"
                ]
                commentary = np.random.choice(fallbacks)
            
            return commentary
            
        except Exception as e:
            print(f"Error generating commentary: {str(e)}")
            return "The scene demonstrates effective use of cinematic techniques to create a compelling viewing experience."

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