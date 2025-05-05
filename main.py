# Disable MPS and CUDA before anything else
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_MPS"] = "0"
os.environ["TORCH_DEVICE"] = "cpu"

import torch
# Explicitly disable CUDA and MPS
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.enabled = False
if hasattr(torch.backends, 'cuda'):
    torch.backends.cuda.enabled = False
torch.set_default_tensor_type('torch.FloatTensor')

# Now import the rest
from video_analyzer import VideoAnalyzer
from video_commentator import VideoCommentator
from ai_processor import AIProcessor
import numpy as np
from typing import Dict

def analyze_video(video_path: str, segments: list):
    """Analyze a video and generate commentaries."""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
        
    print(f"\nAnalyzing video segments from '{os.path.basename(video_path)}'...")
    print("=" * 80)
    
    # Initialize components
    print("Initializing analysis components...")
    try:
        analyzer = VideoAnalyzer()
        commentator = VideoCommentator(video_path)
        ai_processor = AIProcessor()
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        return
    
    # Process each segment
    total_segments = len(segments)
    print(f"\nProcessing {total_segments} video segments...")
    
    successful_segments = 0
    for current_segment, (start_time, end_time) in enumerate(segments, 1):
        try:
            print(f"\nAnalyzing segment {current_segment}/{total_segments} ({start_time}-{end_time} seconds)...")
            print("-" * 40)
            
            print("Extracting video insights...")
            # Get insights for the segment
            analysis_result = analyzer.analyze_segment(video_path, start_time, end_time)
            
            if not analysis_result:
                print("⚠️ Warning: No insights extracted for this segment")
                continue
                
            print("✓ Video insights extracted")
            
            # Generate commentary based on analysis
            print("Generating commentary...")
            commentary = ai_processor.suggest_commentary(analysis_result)
            print("✓ Commentary generated")
            
            print("Adding commentary to collection...")
            # Add commentary with timing and priority
            commentator.add_commentary(
                start_time=start_time,
                end_time=end_time,
                commentary=commentary,
                priority=0.8,  # High priority for main segments
                insights_used=list(analysis_result.keys())
            )
            print("✓ Commentary added")
            successful_segments += 1
            
            print(f"Completed segment {current_segment}/{total_segments}")
            
        except Exception as e:
            print(f"Error processing segment {current_segment}: {str(e)}")
            continue
    
    if successful_segments == 0:
        print("\n⚠️ Warning: No segments were successfully processed")
        return
        
    # Export commentaries to JSON
    print(f"\nExporting commentaries to JSON ({successful_segments}/{total_segments} segments processed)...")
    try:
        output_path = "commentaries.json"
        commentator.export_commentaries(output_path)
        print(f"✓ Commentaries exported to {output_path}")
    except Exception as e:
        print(f"Error exporting commentaries: {str(e)}")

def format_insights_to_text(insights: Dict) -> str:
    """Format insights dictionary into structured text."""
    try:
        insight_text = ""
        
        # Visual Analysis
        if 'visual_analysis' in insights and insights['visual_analysis']:
            visual = insights['visual_analysis']
            insight_text += "visual_analysis:\n"
            if 'color_palette' in visual:
                colors = visual['color_palette']
                if colors:
                    insight_text += f"  color_palette: {', '.join(str(c) for c in colors)}\n"
            if 'brightness_levels' in visual:
                brightness = visual['brightness_levels']
                if brightness is not None:
                    insight_text += f"  brightness_levels: {np.mean(brightness)}\n"
            if 'scene_changes' in visual:
                changes = visual['scene_changes']
                if changes:
                    insight_text += f"  scene_changes: {', '.join(str(c) for c in changes)}\n"
            if 'composition_analysis' in visual:
                composition = visual['composition_analysis']
                if composition:
                    insight_text += "  composition_analysis:\n"
                    if 'rule_of_thirds_scores' in composition:
                        scores = composition['rule_of_thirds_scores']
                        if scores:
                            insight_text += f"    rule_of_thirds_scores: {', '.join(str(s) for s in scores)}\n"
                    if 'symmetry_score' in composition:
                        insight_text += f"    symmetry_score: {composition['symmetry_score']}\n"
        
        # Audio Analysis
        if 'audio_analysis' in insights and insights['audio_analysis']:
            audio = insights['audio_analysis']
            insight_text += "audio_analysis:\n"
            if 'volume_levels' in audio:
                volume = audio['volume_levels']
                if volume is not None:
                    insight_text += f"  volume_levels: {np.mean(volume)}\n"
            if 'pitch_features' in audio:
                pitch = audio['pitch_features']
                if pitch:
                    insight_text += "  pitch_features:\n"
                    if 'f0' in pitch:
                        insight_text += f"    f0: {', '.join(str(p) for p in pitch['f0'])}\n"
                    if 'chroma' in pitch:
                        insight_text += f"    chroma: {', '.join(str(c) for c in pitch['chroma'])}\n"
            if 'tempo' in audio:
                insight_text += f"  tempo: {audio['tempo']}\n"
            if 'spectral_features' in audio:
                spectral = audio['spectral_features']
                if spectral:
                    insight_text += "  spectral_features:\n"
                    for k, v in spectral.items():
                        insight_text += f"    {k}: {v}\n"
        
        # Speech Analysis
        if 'speech_analysis' in insights and insights['speech_analysis']:
            speech = insights['speech_analysis']
            insight_text += "speech_analysis:\n"
            if 'speech_segments' in speech:
                speech_segs = speech['speech_segments']
                if speech_segs:
                    insight_text += "  speech_segments:\n"
                    for seg_idx, seg in enumerate(speech_segs):
                        insight_text += f"    segment_{seg_idx}: start={seg['start']}, end={seg['end']}\n"
            if 'audio_features' in speech:
                features = speech['audio_features']
                if features:
                    insight_text += "  audio_features:\n"
                    for k, v in features.items():
                        insight_text += f"    {k}: {v}\n"
        
        return insight_text
        
    except Exception as e:
        print(f"Error formatting insights: {str(e)}")
        return ""

def main():
    # Video path and segments to analyze
    video_path = "videos/httyd.mp4"
    segments = [
        (0, 150),    # Introduction
        (340, 700),   # Middle segment
        (1200, 1390),   # Later segment
    ]
    
    try:
        analyze_video(video_path, segments)
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main() 