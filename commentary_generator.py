from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import Dict, List, Tuple
from base_knowledge_graph import BaseKnowledgeGraph

class CommentaryGenerator:
    def __init__(self, knowledge_graph_path: str = None):
        """Initialize the commentary generator with necessary models."""
        # Initialize specialized analysis pipelines
        self.style_analyzer = pipeline("text-classification",
                                     model="cross-encoder/nli-distilroberta-base",
                                     model_kwargs={"device_map": "auto"})
        
        # Load base knowledge graph
        self.base_knowledge = BaseKnowledgeGraph.load_from_file(knowledge_graph_path) if knowledge_graph_path else BaseKnowledgeGraph()
        
    def generate_commentary(self, 
                          video_analysis: Dict,
                          segment_duration: float) -> List[Dict]:
        """Generate commentary based on video analysis."""
        # Extract key insights from analysis
        insights = self._extract_insights(video_analysis)
        
        # Generate commentary segments
        commentary_segments = []
        
        # Get optimal commentary points
        commentary_points = video_analysis['commentary_points']
        
        for point in commentary_points:
            # Calculate available time for commentary
            available_time = point['duration']
            
            # Select insights relevant to this time point
            relevant_insights = self._select_relevant_insights(
                insights, 
                point['start_time'],
                video_analysis
            )
            
            # Generate commentary text
            commentary_text = self._generate_commentary_text(
                relevant_insights,
                available_time
            )
            
            commentary_segments.append({
                'text': commentary_text,
                'start_time': point['start_time'],
                'end_time': point['start_time'] + available_time,
                'priority': point['priority'],
                'insights_used': relevant_insights
            })
            
        return commentary_segments
        
    def _extract_insights(self, video_analysis: Dict) -> List[Dict]:
        """Extract key insights from video analysis."""
        insights = []
        
        # Visual insights
        visual = video_analysis['visual_analysis']
        for i, (colors, composition) in enumerate(zip(visual['color_palette'], 
                                                    visual['composition_analysis'])):
            if i in visual['scene_changes']:
                insights.append({
                    'type': 'scene_change',
                    'time': i,
                    'colors': colors,
                    'composition': composition
                })
            
            # Check for significant composition elements
            if max(composition['rule_of_thirds_scores']) > 0.7:
                insights.append({
                    'type': 'composition',
                    'time': i,
                    'focus_score': max(composition['rule_of_thirds_scores']),
                    'symmetry': composition['symmetry_score']
                })
                
        # Audio insights
        audio = video_analysis['audio_analysis']
        spectral_contrast = audio['spectral_features']['contrast']
        for i, volume in enumerate(audio['volume_levels']):
            if volume > np.mean(audio['volume_levels']) + np.std(audio['volume_levels']):
                insights.append({
                    'type': 'audio_peak',
                    'time': i,
                    'volume': float(volume),
                    'spectral_contrast': float(spectral_contrast[min(i, len(spectral_contrast)-1)])
                })
                
        # Add pitch insights
        if 'pitch_features' in audio:
            f0 = audio['pitch_features']['f0']
            chroma = audio['pitch_features']['chroma']
            for i, pitch in enumerate(f0):
                if pitch > 0:  # Valid pitch detected
                    insights.append({
                        'type': 'pitch',
                        'time': i,
                        'frequency': float(pitch),
                        'chroma_value': float(chroma[min(i, len(chroma)-1)])
                    })
                
        # Speech insights
        for segment in video_analysis['speech_analysis']['speech_segments']:
            insights.append({
                'type': 'speech',
                'text': segment['text'],
                'emotion': segment['emotion']
            })
            
        return insights
        
    def _select_relevant_insights(self, 
                                insights: List[Dict],
                                time_point: float,
                                video_analysis: Dict) -> List[Dict]:
        """Select insights relevant to a specific time point."""
        relevant = []
        seen_types = set()  # Track seen insight types to avoid duplicates
        
        for insight in insights:
            # Only take one insight of each type unless it's significantly different
            if insight['type'] in seen_types:
                continue
                
            # Check if insight is within time window
            if 'time' in insight:
                time_window = 2.0  # Look at insights within 2 seconds
                if abs(insight['time'] - time_point) <= time_window:
                    relevant.append(insight)
                    seen_types.add(insight['type'])
            else:
                # For insights without specific time (like general speech analysis)
                relevant.append(insight)
                seen_types.add(insight['type'])
                
        return relevant
        
    def _generate_commentary_text(self, 
                                insights: List[Dict],
                                available_time: float) -> str:
        """Generate commentary text based on insights."""
        # Group insights by type
        grouped_insights = {}
        for insight in insights:
            if insight['type'] not in grouped_insights:
                grouped_insights[insight['type']] = []
            grouped_insights[insight['type']].append(insight)
        
        # Build commentary parts
        commentary_parts = []
        
        # Add visual insights
        visual_parts = []
        if 'scene_change' in grouped_insights:
            scene = grouped_insights['scene_change'][0]
            colors = self._describe_colors(scene['colors'])
            visual_parts.append(f"using {colors}")
            
        if 'composition' in grouped_insights:
            comp = grouped_insights['composition'][0]
            composition = self._describe_composition({
                'rule_of_thirds_scores': [comp['focus_score']], 
                'symmetry_score': comp['symmetry']
            })
            visual_parts.append(f"with {composition}")
            
        if visual_parts:
            commentary_parts.append("The shot is composed " + " ".join(visual_parts))
            
        # Add audio insights
        if 'audio_peak' in grouped_insights:
            peaks = grouped_insights['audio_peak']
            if len(peaks) > 2:
                commentary_parts.append("The audio builds in intensity")
            else:
                commentary_parts.append("There's a notable emphasis in the audio")
                
        # Add speech insights
        if 'speech' in grouped_insights:
            speech = grouped_insights['speech'][0]
            text = speech['text'].strip()
            if text:
                commentary_parts.append(
                    f"The speaker conveys {speech['emotion']} as they explain: '{text}'"
                )
            
        # Combine parts into coherent commentary
        if commentary_parts:
            commentary = " ".join(commentary_parts)
        else:
            commentary = "The scene continues with subtle visual and audio elements"
            
        return commentary
        
    def _describe_colors(self, colors: List[Tuple[int, int, int]]) -> str:
        """Convert RGB colors to descriptive text."""
        color_names = []
        for r, g, b in colors:
            if r > g and r > b:
                color_names.append("warm reddish tones")
            elif g > r and g > b:
                color_names.append("natural green hues")
            elif b > r and b > g:
                color_names.append("cool blue shades")
            else:
                brightness = (r + g + b) / 3
                if brightness > 200:
                    color_names.append("bright highlights")
                elif brightness < 50:
                    color_names.append("deep shadows")
                else:
                    color_names.append("balanced midtones")
                    
        return " and ".join(color_names[:2])
        
    def _describe_composition(self, composition: Dict) -> str:
        """Convert composition analysis to descriptive text."""
        if composition['symmetry_score'] > 0.8:
            return "striking symmetry"
        elif max(composition['rule_of_thirds_scores']) > 0.7:
            return "strong focal points following the rule of thirds"
        else:
            return "dynamic asymmetrical balance"
        
    def _select_best_response(self, 
                            responses: List[Dict],
                            insights: List[Dict]) -> str:
        """Select the most appropriate response based on insights."""
        best_score = -1
        best_response = ""
        
        for response in responses:
            text = response['generated_text']
            
            # Score response based on relevance to insights
            score = 0
            for insight in insights:
                if insight['type'] in text.lower():
                    score += 1
                if 'emotion' in insight and insight['emotion'].lower() in text.lower():
                    score += 2
                    
            # Check style appropriateness
            style_score = self.style_analyzer(text)[0]['score']
            score *= style_score
            
            if score > best_score:
                best_score = score
                best_response = text
                
        return best_response
        
    def _optimize_timing(self, 
                        point: Dict,
                        commentary_text: str,
                        video_analysis: Dict) -> Dict:
        """Optimize the timing of commentary delivery."""
        words = len(commentary_text.split())
        required_time = words / 2.5  # Assuming 2.5 words per second
        
        # Adjust start time to avoid overlapping with important moments
        start_time = point['start_time']
        end_time = start_time + required_time
        
        # Check for speech overlap
        for segment in video_analysis['speech_analysis']['speech_segments']:
            if 'time' in segment:
                # Adjust timing to avoid overlap with important dialogue
                if (start_time <= segment['time'] <= end_time and 
                    segment.get('emotion') in ['angry', 'excited', 'surprised']):
                    start_time = segment['time'] + 1.0  # Add buffer
                    end_time = start_time + required_time
                    
        return {
            'start': start_time,
            'end': end_time
        } 