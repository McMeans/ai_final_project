import networkx as nx
import json
from typing import List, Tuple
import os

class BaseKnowledgeGraph:
    def __init__(self):
        """Initialize the base knowledge graph with predefined concepts and relationships."""
        self.graph = nx.Graph()
        self._initialize_concepts()
        
    def _initialize_concepts(self):
        """Initialize the knowledge graph with predefined concepts and relationships."""
        # Core concept categories
        concepts = {
            'visual_elements': [
                'lighting', 'composition', 'color_scheme', 'camera_angle',
                'camera_movement', 'framing', 'focus', 'depth_of_field'
            ],
            'audio_elements': [
                'dialogue', 'background_music', 'sound_effects', 'ambient_sound',
                'voice_over', 'audio_mixing', 'volume_levels'
            ],
            'narrative_elements': [
                'plot_point', 'character_development', 'story_arc',
                'exposition', 'conflict', 'resolution', 'pacing'
            ],
            'technical_aspects': [
                'video_quality', 'editing_technique', 'transition_type',
                'special_effects', 'color_grading', 'post_processing'
            ],
            'emotional_aspects': [
                'mood', 'tension', 'emotional_impact', 'atmosphere',
                'viewer_engagement', 'dramatic_effect'
            ]
        }
        
        # Add all concepts as nodes
        for category, elements in concepts.items():
            # Add category node
            self.graph.add_node(category, type='category')
            # Add elements and connect to category
            for element in elements:
                self.graph.add_node(element, type='concept')
                self.graph.add_edge(category, element, relation='contains')
        
        # Add relationships between concepts
        relationships = [
            ('lighting', 'mood', 'influences'),
            ('camera_angle', 'emotional_impact', 'affects'),
            ('sound_effects', 'atmosphere', 'creates'),
            ('background_music', 'emotional_impact', 'enhances'),
            ('pacing', 'tension', 'controls'),
            ('color_scheme', 'mood', 'sets'),
            ('editing_technique', 'pacing', 'determines'),
            ('dialogue', 'character_development', 'reveals'),
            ('camera_movement', 'dramatic_effect', 'amplifies'),
            ('special_effects', 'viewer_engagement', 'increases')
        ]
        
        # Add relationships
        for source, target, relation in relationships:
            self.graph.add_edge(source, target, relation=relation)
            
        # Add common descriptors
        descriptors = {
            'quality': ['excellent', 'good', 'average', 'poor'],
            'intensity': ['high', 'medium', 'low'],
            'complexity': ['simple', 'moderate', 'complex'],
            'effectiveness': ['effective', 'neutral', 'ineffective']
        }
        
        for category, values in descriptors.items():
            self.graph.add_node(category, type='descriptor_category')
            for value in values:
                self.graph.add_node(value, type='descriptor')
                self.graph.add_edge(category, value, relation='has_value')
    
    def add_concept(self, concept: str, concept_type: str, relationships: List[Tuple[str, str]] = None):
        """Add a new concept to the knowledge graph with optional relationships."""
        self.graph.add_node(concept, type=concept_type)
        if relationships:
            for related_concept, relation in relationships:
                if related_concept in self.graph:
                    self.graph.add_edge(concept, related_concept, relation=relation)
    
    def get_related_concepts(self, concept: str, relation_type: str = None) -> List[str]:
        """Get concepts related to the given concept, optionally filtered by relation type."""
        if concept not in self.graph:
            return []
            
        related = []
        for neighbor in self.graph.neighbors(concept):
            if relation_type is None or self.graph[concept][neighbor].get('relation') == relation_type:
                related.append(neighbor)
        return related
    
    def get_concept_type(self, concept: str) -> str:
        """Get the type of a concept."""
        return self.graph.nodes[concept].get('type') if concept in self.graph else None
    
    def save_to_file(self, filepath: str):
        """Save the knowledge graph to a JSON file."""
        data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': [(u, v, d) for u, v, d in self.graph.edges(data=True)]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'BaseKnowledgeGraph':
        """Load a knowledge graph from a JSON file."""
        instance = cls()
        if not os.path.exists(filepath):
            return instance
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        instance.graph = nx.Graph()
        # Add nodes
        for node, attrs in data['nodes'].items():
            instance.graph.add_node(node, **attrs)
        # Add edges
        for u, v, d in data['edges']:
            instance.graph.add_edge(u, v, **d)
            
        return instance 