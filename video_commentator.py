# System imports
import json
from typing import Dict, List, Tuple
from datetime import timedelta

# Third-party imports
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

# Local imports
from ai_processor import AIProcessor

class VideoCommentator:
    def __init__(self, video_path: str, knowledge_graph_path: str = None):
        """Initialize the video commentator with a video file."""
        self.video_path = video_path
        self.video = VideoFileClip(video_path)
        self.commentaries = []
        self.knowledge_graph = nx.Graph()
        self.ai_processor = AIProcessor(knowledge_graph_path)
        
        # MDP state variables
        self.mdp_state = {
            'last_commentary_end': 0.0,
            'scene_intensity': 0.0,
            'dialogue_presence': 0.0,
            'visual_complexity': 0.0
        }
        
    def load_commentaries(self, json_path: str) -> None:
        """Load commentaries from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.commentaries = data['commentaries']
            
    def _calculate_mdp_reward(self, state: Dict, action: Dict) -> float:
        """Calculate reward for MDP-based commentary timing."""
        reward = 0.0
        
        # Reward for not interrupting important moments
        if state['scene_intensity'] > 0.7 and action['speak']:
            reward -= 10.0
            
        # Reward for maintaining good spacing between commentaries
        time_since_last = action['start_time'] - state['last_commentary_end']
        if time_since_last < 5.0:  # Too close to previous commentary
            reward -= 5.0
        elif time_since_last > 30.0:  # Too long without commentary
            reward -= (time_since_last - 30.0) * 0.1
            
        # Reward for relevant insights
        if action['insights_used']:
            reward += len(action['insights_used']) * 2.0
            
        # Reward for avoiding dialogue
        if state['dialogue_presence'] > 0.5 and action['speak']:
            reward -= 8.0
            
        # Reward for matching visual complexity
        if state['visual_complexity'] > 0.7 and action['speak']:
            reward -= 6.0
            
        return reward
        
    def _update_mdp_state(self, analysis: Dict, time: float):
        """Update MDP state based on current analysis."""
        # Update scene intensity
        self.mdp_state['scene_intensity'] = self._calculate_scene_intensity(analysis)
        
        # Update dialogue presence
        self.mdp_state['dialogue_presence'] = self._calculate_dialogue_presence(analysis)
        
        # Update visual complexity
        self.mdp_state['visual_complexity'] = self._calculate_visual_complexity(analysis)
        
    def _calculate_scene_intensity(self, analysis: Dict) -> float:
        """Calculate current scene intensity based on various factors."""
        intensity = 0.0
        
        # Audio intensity
        if 'audio_analysis' in analysis:
            audio = analysis['audio_analysis']
            if 'volume_levels' in audio:
                intensity += np.mean(audio['volume_levels']) * 0.3
                
        # Visual intensity
        if 'visual_analysis' in analysis:
            visual = analysis['visual_analysis']
            if 'camera_movement' in visual:
                movement = visual['camera_movement']
                if 'movement_scores' in movement:
                    intensity += np.mean(movement['movement_scores']) * 0.3
                    
        # Emotional intensity
        if 'speech_analysis' in analysis:
            speech = analysis['speech_analysis']
            if 'emotion_scores' in speech:
                intensity += np.mean(speech['emotion_scores']) * 0.4
                
        return float(np.clip(intensity, 0.0, 1.0))
        
    def _calculate_dialogue_presence(self, analysis: Dict) -> float:
        """Calculate presence of dialogue in the scene."""
        if 'speech_analysis' not in analysis:
            return 0.0
            
        speech = analysis['speech_analysis']
        if 'speech_segments' not in speech:
            return 0.0
            
        segments = speech['speech_segments']
        if not segments:
            return 0.0
            
        # Calculate percentage of time with dialogue
        total_duration = sum(seg['end'] - seg['start'] for seg in segments)
        return float(np.clip(total_duration / 10.0, 0.0, 1.0))  # Normalize to 10-second window
        
    def _calculate_visual_complexity(self, analysis: Dict) -> float:
        """Calculate visual complexity of the scene."""
        if 'visual_analysis' not in analysis:
            return 0.0
            
        visual = analysis['visual_analysis']
        complexity = 0.0
        
        # Composition complexity
        if 'composition_analysis' in visual:
            comp = visual['composition_analysis']
            if 'rule_of_thirds_scores' in comp:
                complexity += np.std(comp['rule_of_thirds_scores']) * 0.3
            if 'symmetry_scores' in comp:
                complexity += np.std(comp['symmetry_scores']) * 0.3
                
        # Movement complexity
        if 'camera_movement' in visual:
            movement = visual['camera_movement']
            if 'movement_scores' in movement:
                complexity += np.std(movement['movement_scores']) * 0.4
                
        return float(np.clip(complexity, 0.0, 1.0))
        
    def add_commentary(self, start_time: float, end_time: float, commentary: str, priority: float = 0.5, insights_used: List[str] = None) -> None:
        """Add a commentary for a specific video segment using MDP-based timing."""
        if start_time >= end_time:
            raise ValueError("Start time must be less than end time")
        if start_time < 0 or end_time > self.video.duration:
            raise ValueError("Time stamps must be within video duration")
            
        # Calculate MDP reward for this commentary
        action = {
            'speak': True,
            'start_time': start_time,
            'end_time': end_time,
            'insights_used': insights_used or []
        }
        
        reward = self._calculate_mdp_reward(self.mdp_state, action)
        
        # Create a new commentary entry
        commentary_entry = {
            'start_time': start_time,
            'end_time': end_time,
            'text': commentary,
            'priority': priority,
            'insights_used': insights_used or [],
            'mdp_reward': reward
        }
        
        # Add to commentaries list
        self.commentaries.append(commentary_entry)
        
        # Update MDP state
        self.mdp_state['last_commentary_end'] = end_time
        
    def get_commentaries(self) -> List[Dict]:
        """Get all commentaries."""
        return self.commentaries
    
    def _update_knowledge_graph(self, commentary: str, time_range: Tuple[float, float]) -> None:
        """Update the knowledge graph using AI-powered analysis."""
        graph_data = self.ai_processor.generate_knowledge_graph_data(commentary, time_range)
        
        # Add nodes and edges from the analysis
        for triple in graph_data['triples']:
            subject, verb, obj = triple
            # Add nodes with timestamps and types
            self.knowledge_graph.add_node(subject, type='subject', timestamps=[time_range])
            self.knowledge_graph.add_node(verb, type='action', timestamps=[time_range])
            self.knowledge_graph.add_node(obj, type='object', timestamps=[time_range])
            
            # Add edges with relationships
            self.knowledge_graph.add_edge(subject, verb, relation='performs')
            self.knowledge_graph.add_edge(verb, obj, relation='affects')
        
        # Add relevant concepts from base knowledge
        for concept_data in graph_data['relevant_concepts']:
            category = concept_data['category']
            concepts = concept_data['concepts']
            confidence = concept_data['confidence']
            
            # Add category node if it doesn't exist
            if category not in self.knowledge_graph:
                self.knowledge_graph.add_node(category, type='category')
            
            # Add concepts and connect to category
            for concept in concepts:
                if concept not in self.knowledge_graph:
                    self.knowledge_graph.add_node(concept, 
                                                type='concept',
                                                timestamps=[time_range],
                                                confidence=confidence)
                    self.knowledge_graph.add_edge(category, concept, 
                                                relation='contains',
                                                timestamp=time_range)
        
        # Add descriptors
        for descriptor in graph_data['descriptors']:
            if descriptor not in self.knowledge_graph:
                self.knowledge_graph.add_node(descriptor, 
                                            type='descriptor',
                                            timestamps=[time_range])
                
    def visualize_knowledge_graph(self, output_path: str = 'knowledge_graph.png') -> None:
        """Visualize and save the AI-enhanced knowledge graph."""
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.knowledge_graph, k=1, iterations=50)
        
        # Draw nodes with different colors based on type
        node_colors = []
        node_sizes = []
        for node in self.knowledge_graph.nodes():
            node_type = self.knowledge_graph.nodes[node].get('type', 'default')
            # Set color based on node type
            if node_type == 'category':
                node_colors.append('lightblue')
                node_sizes.append(3000)
            elif node_type == 'concept':
                node_colors.append('lightgreen')
                node_sizes.append(2000)
            elif node_type == 'subject':
                node_colors.append('lightpink')
                node_sizes.append(2000)
            elif node_type == 'action':
                node_colors.append('yellow')
                node_sizes.append(1500)
            elif node_type == 'descriptor':
                node_colors.append('orange')
                node_sizes.append(1500)
            else:
                node_colors.append('gray')
                node_sizes.append(1500)
                
        # Draw the graph
        nx.draw(self.knowledge_graph, pos, 
                with_labels=True,
                node_color=node_colors,
                node_size=node_sizes,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                width=1,
                alpha=0.7)
        
        # Add title and legend
        plt.title("AI-Enhanced Knowledge Graph with Base Knowledge", pad=20)
        
        # Add timestamp annotations
        for node, (x, y) in pos.items():
            timestamps = self.knowledge_graph.nodes[node].get('timestamps', [])
            if timestamps:
                timestamp = timestamps[-1]  # Get most recent timestamp
                plt.annotate(f"{timestamp[0]:.1f}s-{timestamp[1]:.1f}s",
                           (x, y),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           va='bottom',
                           fontsize=6)
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    def export_commentaries(self, output_path: str):
        """Export commentaries to a JSON file."""
        # Create export data preserving all commentaries
        export_data = []
        for commentary in self.commentaries:
            export_entry = {
                'start_time': commentary['start_time'],
                'end_time': commentary['end_time'],
                'text': commentary['text']
            }
            export_data.append(export_entry)
            
        # Save to JSON file with proper formatting
        with open(output_path, 'w') as f:
            json.dump({'commentaries': export_data}, f, indent=2)
            
        print(f"Commentaries exported to {output_path}")
        print(f"Exported {len(export_data)} commentaries")
            
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to human-readable timestamp."""
        return str(timedelta(seconds=seconds))
    
    def get_segment_preview(self, start_time: float, end_time: float) -> str:
        """Get a text preview of a video segment with its AI-enhanced commentary."""
        if (start_time, end_time) not in self.commentaries:
            return f"No commentary for segment {self.format_timestamp(start_time)} - {self.format_timestamp(end_time)}"
        
        commentary = self.commentaries[(start_time, end_time)]
        analysis = self.ai_processor.analyze_segment(commentary)
        
        # Create a detailed preview with base knowledge insights
        preview_parts = [
            f"\nSegment: {self.format_timestamp(start_time)} - {self.format_timestamp(end_time)}",
            f"Commentary: {commentary}",
            f"Summary: {analysis['summary'] if analysis['summary'] else 'N/A'}",
            f"Sentiment: {analysis['sentiment']['label']} ({analysis['sentiment']['score']:.2f})",
            f"Key Elements: {', '.join(f'{ent[0]} ({ent[1]})' for ent in analysis['entities']) if analysis['entities'] else 'None detected'}"
        ]
        
        # Add relevant concepts from base knowledge
        if analysis['relevant_concepts']:
            concepts_text = "Relevant Concepts:\n" + "\n".join(
                f"- {concept['category']} (confidence: {concept['confidence']:.2f}): {', '.join(concept['concepts'][:3])}"
                for concept in analysis['relevant_concepts']
            )
            preview_parts.append(concepts_text)
            
        return "\n".join(preview_parts) 