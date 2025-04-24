from video_commentator import VideoCommentator
from base_knowledge_graph import BaseKnowledgeGraph
from video_analyzer import VideoAnalyzer
from commentary_generator import CommentaryGenerator
from script_processor import ScriptProcessor
from typing import List, Tuple, Dict

def analyze_video(video_path: str, segments: List[Tuple[float, float]], script_path: str = None):
    """Analyze specific video segments with optional script analysis."""
    # Initialize components
    print("Initializing analysis components...")
    base_knowledge = BaseKnowledgeGraph()
    base_knowledge.save_to_file('base_knowledge.json')
    
    video_analyzer = VideoAnalyzer()
    commentary_generator = CommentaryGenerator('base_knowledge.json')
    commentator = VideoCommentator(video_path, knowledge_graph_path='base_knowledge.json')
    
    # Process script if provided
    script_processor = None
    script_analysis = None
    if script_path:
        print("\nProcessing script...")
        script_processor = ScriptProcessor()
        with open(script_path, 'r') as f:
            script_text = f.read()
        script_analysis = script_processor.process_script(script_text)
    
    # Process each segment
    print("\nProcessing video segments...")
    for start_time, end_time in segments:
        print(f"\nAnalyzing segment {start_time:.1f}-{end_time:.1f} seconds...")
        print("-" * 40)
        
        # Analyze the video segment
        analysis = video_analyzer.analyze_segment(video_path, start_time, end_time)
        
        # If script is available, enhance analysis with script information
        if script_processor and script_analysis:
            # Find corresponding script sections
            scene_info = find_matching_scenes(script_analysis, start_time, end_time)
            if scene_info:
                analysis['script_analysis'] = scene_info
        
        # Generate AI commentary
        commentaries = commentary_generator.generate_commentary(
            analysis,
            end_time - start_time
        )
        
        # Add each commentary with its optimal timing
        for commentary in commentaries:
            commentator.add_commentary(
                start_time=commentary['start_time'],
                end_time=commentary['end_time'],
                commentary=commentary['text'],
                auto_enhance=False  # Already enhanced by AI
            )
            
            print(f"\nCommentary at {commentary['start_time']:.1f}-{commentary['end_time']:.1f}:")
            print(commentary['text'])
            print(f"Priority: {commentary['priority']}")
            print("Based on:", ", ".join(insight['type'] for insight in commentary['insights_used']))
    
    # Generate and save the AI-enhanced knowledge graph
    print("\nGenerating AI-enhanced knowledge graph...")
    commentator.visualize_knowledge_graph('ai_knowledge_graph.png')
    print("Knowledge graph has been saved to 'ai_knowledge_graph.png'")
    
    # Export commentaries with AI analysis
    print("\nExporting commentaries with AI analysis...")
    commentator.export_commentaries('ai_commentaries.json')
    print("Commentaries and analysis have been exported to 'ai_commentaries.json'")
    print("\nAnalysis complete!")

def find_matching_scenes(script_analysis: Dict, start_time: float, end_time: float) -> Dict:
    """Find scenes in the script that match the time segment."""
    # This is a placeholder - in a real implementation, you would need
    # to align the script with the video timeline
    return script_analysis['scenes'][0] if script_analysis['scenes'] else None

def main():
    # Video file path
    video_path = "videos/How To Make Garlic Bread.mp4"
    
    # Define distinct segments to analyze (in seconds)
    segments = [
        (300.0, 415.0),    # Introduction/opening
        (589.0, 700.0),   # Middle segment
        (1000.0, 1100.0)    # Later segment
    ]
    
    print("\nAnalyzing video segments from 'How To Make Garlic Bread'...")
    print("=" * 80)
    
    # Run the analysis
    analyze_video(video_path, segments)

if __name__ == "__main__":
    main() 