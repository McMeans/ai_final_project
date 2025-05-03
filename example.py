import os
from video_analyzer import VideoAnalyzer
from video_commentator import VideoCommentator
from ai_processor import AIProcessor
import json

def analyze_video(video_path: str, segments: list):
    """Analyze a video and generate commentaries."""
    print(f"\nAnalyzing video segments from '{os.path.basename(video_path)}'...")
    print("=" * 80)
    
    # Initialize components
    print("Initializing analysis components...")
    analyzer = VideoAnalyzer()
    commentator = VideoCommentator(video_path)
    ai_processor = AIProcessor()
    
    # Process each segment
    print("\nProcessing video segments...")
    for i, (start_time, end_time) in enumerate(segments, 1):
        print(f"\nAnalyzing segment {i}/{len(segments)} ({start_time}-{end_time} seconds)...")
        print("-" * 40)
        
        print("Extracting video insights...")
        # Get insights for the segment
        insights = analyzer.analyze_segment(video_path, start_time, end_time)
        print("✓ Video insights extracted")
        
        print("Converting insights to text...")
        # Convert insights to text format
        insight_text = ""
        for key, value in insights.items():
            if isinstance(value, dict):
                insight_text += f"{key}: {', '.join(f'{k}: {v}' for k, v in value.items())}\n"
            else:
                insight_text += f"{key}: {value}\n"
        print("✓ Insights converted to text")
        
        print("Generating commentary...")
        # Generate commentary using AI
        commentary = ai_processor.suggest_commentary(insight_text)
        print("✓ Commentary generated")
        
        print("Adding commentary to collection...")
        # Add commentary with timing and priority
        commentator.add_commentary(
            start_time=start_time,
            end_time=end_time,
            commentary=commentary,
            priority=0.8,  # High priority for main segments
            insights_used=list(insights.keys())
        )
        print("✓ Commentary added")
        print(f"Completed segment {i}/{len(segments)}")
    
    # Export commentaries to JSON
    print("\nExporting commentaries to JSON...")
    output_path = "commentaries.json"
    commentator.export_commentaries(output_path)
    print(f"✓ Commentaries exported to {output_path}")

def main():
    # Video path and segments to analyze
    video_path = "videos/httyd.mp4"
    segments = [
        (0, 150),    # Introduction
        (340, 700),   # Middle segment
        (1200, 1390),   # Later segment
    ]
    
    analyze_video(video_path, segments)

if __name__ == "__main__":
    main() 