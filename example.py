import os
from video_analyzer import VideoAnalyzer
from video_commentator import VideoCommentator
from ai_processor import AIProcessor
import json
import numpy as np

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
    total_segments = len(segments)
    print(f"\nProcessing {total_segments} video segments...")
    
    for current_segment, (start_time, end_time) in enumerate(segments, 1):
        print(f"\nAnalyzing segment {current_segment}/{total_segments} ({start_time}-{end_time} seconds)...")
        print("-" * 40)
        
        print("Extracting video insights...")
        # Get insights for the segment
        insights = analyzer.analyze_segment(video_path, start_time, end_time)
        print("✓ Video insights extracted")
        
        print("Converting insights to text...")
        # Convert insights to text format, preserving all analysis details
        insight_text = ""
        
        # Visual Analysis
        visual = insights['visual_analysis']
        insight_text += "visual_analysis:\n"
        if 'color_palette' in visual:
            colors = visual['color_palette']
            insight_text += f"  color_palette: {', '.join(str(c) for c in colors)}\n"
        if 'brightness_levels' in visual:
            brightness = visual['brightness_levels']
            insight_text += f"  brightness_levels: {np.mean(brightness)}\n"
        if 'scene_changes' in visual:
            changes = visual['scene_changes']
            insight_text += f"  scene_changes: {', '.join(str(c) for c in changes)}\n"
        if 'composition_analysis' in visual:
            composition = visual['composition_analysis']
            insight_text += "  composition_analysis:\n"
            if 'rule_of_thirds_scores' in composition:
                scores = composition['rule_of_thirds_scores']
                insight_text += f"    rule_of_thirds_scores: {', '.join(str(s) for s in scores)}\n"
            if 'symmetry_score' in composition:
                insight_text += f"    symmetry_score: {composition['symmetry_score']}\n"
        
        # Audio Analysis
        audio = insights['audio_analysis']
        insight_text += "audio_analysis:\n"
        if 'volume_levels' in audio:
            volume = audio['volume_levels']
            insight_text += f"  volume_levels: {np.mean(volume)}\n"
        if 'pitch_features' in audio:
            pitch = audio['pitch_features']
            insight_text += "  pitch_features:\n"
            if 'f0' in pitch:
                insight_text += f"    f0: {', '.join(str(p) for p in pitch['f0'])}\n"
            if 'chroma' in pitch:
                insight_text += f"    chroma: {', '.join(str(c) for c in pitch['chroma'])}\n"
        if 'tempo' in audio:
            insight_text += f"  tempo: {audio['tempo']}\n"
        if 'spectral_features' in audio:
            spectral = audio['spectral_features']
            insight_text += "  spectral_features:\n"
            for k, v in spectral.items():
                insight_text += f"    {k}: {v}\n"
        
        # Speech Analysis
        speech = insights['speech_analysis']
        insight_text += "speech_analysis:\n"
        if 'speech_segments' in speech:
            speech_segs = speech['speech_segments']
            insight_text += "  speech_segments:\n"
            for seg_idx, seg in enumerate(speech_segs):
                insight_text += f"    segment_{seg_idx}: start={seg['start']}, end={seg['end']}\n"
        if 'audio_features' in speech:
            features = speech['audio_features']
            insight_text += "  audio_features:\n"
            for k, v in features.items():
                insight_text += f"    {k}: {v}\n"
        
        # Segment Characteristics
        characteristics = insights['segment_characteristics']
        insight_text += "segment_characteristics:\n"
        if 'dominant_visual_elements' in characteristics:
            elements = characteristics['dominant_visual_elements']
            insight_text += "  dominant_visual_elements:\n"
            if 'colors' in elements:
                insight_text += "    colors:\n"
                for color, count in elements['colors']:
                    insight_text += f"      {color}: {count}\n"
            if 'composition' in elements:
                insight_text += "    composition:\n"
                for comp, count in elements['composition']:
                    insight_text += f"      {comp}: {count}\n"
        
        if 'audio_patterns' in characteristics:
            patterns = characteristics['audio_patterns']
            insight_text += "  audio_patterns:\n"
            if 'volume_changes' in patterns:
                changes = patterns['volume_changes']
                insight_text += f"    volume_changes: max_increase={changes['max_increase']}, max_decrease={changes['max_decrease']}, average_change={changes['average_change']}\n"
            if 'pitch_movements' in patterns:
                pitch = patterns['pitch_movements']
                insight_text += f"    pitch_movements: range={pitch['range']}, average={pitch['average']}, variability={pitch['variability']}\n"
        
        if 'emotional_arc' in characteristics:
            arc = characteristics['emotional_arc']
            insight_text += "  emotional_arc:\n"
            if 'start_emotion' in arc:
                insight_text += f"    start_emotion: {arc['start_emotion']}\n"
            if 'end_emotion' in arc:
                insight_text += f"    end_emotion: {arc['end_emotion']}\n"
            if 'climax_points' in arc:
                insight_text += f"    climax_points: {', '.join(str(p) for p in arc['climax_points'])}\n"
        
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
        print(f"Completed segment {current_segment}/{total_segments}")
    
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