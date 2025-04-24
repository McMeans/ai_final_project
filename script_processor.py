import re
from typing import List, Dict, Tuple
from datetime import timedelta

class ScriptProcessor:
    def __init__(self):
        """Initialize the script processor."""
        self.time_pattern = re.compile(r'(\d{1,2}:\d{2}:\d{2})')
        self.dialogue_pattern = re.compile(r'([A-Z\s]+):\s*(.*)')
        
    def parse_script(self, script_text: str) -> List[Dict]:
        """Parse a script text into structured segments."""
        segments = []
        current_time = None
        current_dialogue = []
        
        for line in script_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for timestamp
            time_match = self.time_pattern.search(line)
            if time_match:
                # Save previous segment if exists
                if current_time and current_dialogue:
                    segments.append({
                        'time': current_time,
                        'dialogue': '\n'.join(current_dialogue)
                    })
                current_time = self._parse_timestamp(time_match.group(1))
                current_dialogue = []
                continue
                
            # Check for dialogue
            dialogue_match = self.dialogue_pattern.match(line)
            if dialogue_match:
                speaker, text = dialogue_match.groups()
                current_dialogue.append(f"{speaker}: {text}")
                
        # Add the last segment if exists
        if current_time and current_dialogue:
            segments.append({
                'time': current_time,
                'dialogue': '\n'.join(current_dialogue)
            })
            
        return segments
        
    def _parse_timestamp(self, time_str: str) -> float:
        """Convert timestamp string to seconds."""
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
        
    def find_commentary_points(self, segments: List[Dict], min_gap: float = 5.0) -> List[Tuple[float, float]]:
        """Find suitable points for commentary insertion."""
        points = []
        
        for i in range(len(segments) - 1):
            current_time = segments[i]['time']
            next_time = segments[i + 1]['time']
            gap = next_time - current_time
            
            if gap >= min_gap:
                # Add a point in the middle of the gap
                start = current_time + (gap * 0.3)  # 30% into the gap
                end = current_time + (gap * 0.7)    # 70% into the gap
                points.append((start, end))
                
        return points
        
    def generate_commentary_suggestions(self, segments: List[Dict]) -> List[Dict]:
        """Generate commentary suggestions based on script content."""
        suggestions = []
        
        for i, segment in enumerate(segments):
            dialogue = segment['dialogue']
            time = segment['time']
            
            # Basic analysis of dialogue
            num_speakers = len(set(line.split(':')[0] for line in dialogue.split('\n')))
            dialogue_length = len(dialogue)
            
            # Generate suggestion based on content
            if num_speakers > 1:
                suggestion = {
                    'time': time,
                    'type': 'dialogue_analysis',
                    'content': f"Multiple speakers engaged in conversation"
                }
            elif dialogue_length > 100:
                suggestion = {
                    'time': time,
                    'type': 'monologue',
                    'content': "Extended monologue with detailed explanation"
                }
            else:
                suggestion = {
                    'time': time,
                    'type': 'brief_dialogue',
                    'content': "Brief exchange between characters"
                }
                
            suggestions.append(suggestion)
            
        return suggestions 