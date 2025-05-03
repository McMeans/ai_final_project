import os
from typing import Dict, List, Tuple
from moviepy.editor import VideoFileClip
import networkx as nx
import matplotlib.pyplot as plt
import json
from datetime import timedelta
from ai_processor import AIProcessor

class VideoCommentator:
    def __init__(self, video_path: str, knowledge_graph_path: str = None):
        """Initialize the video commentator with a video file."""
        self.video_path = video_path
        self.video = VideoFileClip(video_path)
        self.commentaries = []
        self.knowledge_graph = nx.Graph()
        self.ai_processor = AIProcessor(knowledge_graph_path)
        
    def load_commentaries(self, json_path: str) -> None:
        """Load commentaries from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.commentaries = data['commentaries']
            
    def add_commentary(self, start_time: float, end_time: float, commentary: str, priority: float = 0.5, insights_used: List[str] = None) -> None:
        """Add a commentary for a specific video segment."""
        if start_time >= end_time:
            raise ValueError("Start time must be less than end time")
        if start_time < 0 or end_time > self.video.duration:
            raise ValueError("Time stamps must be within video duration")
            
        self.commentaries.append({
            'start_time': start_time,
            'end_time': end_time,
            'text': commentary,
            'priority': priority,
            'insights_used': insights_used or []
        })
        
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
        # Create simplified export data
        export_data = []
        for commentary in self.commentaries:
            export_entry = {
                'start_time': commentary['start_time'],
                'end_time': commentary['end_time'],
                'text': commentary['text']
            }
            export_data.append(export_entry)
            
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump({'commentaries': export_data}, f, indent=2)
            
        print(f"Commentaries exported to {output_path}")
            
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