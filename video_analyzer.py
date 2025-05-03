import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
import librosa
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import speech_recognition as sr
from typing import Dict, List, Tuple
import tempfile
import os

class VideoAnalyzer:
    def __init__(self):
        """Initialize the video analyzer with necessary models."""
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize scene detection model
        self.scene_detector = pipeline("zero-shot-image-classification",
                                     model="openai/clip-vit-base-patch32")
        
        # Initialize speech detection model
        self.audio_classifier = pipeline("audio-classification",
                                       model="superb/hubert-base-superb-sid")
        
        # Initialize emotion recognition for speech
        self.emotion_recognizer = pipeline("text-classification",
                                         model="j-hartmann/emotion-english-distilroberta-base")
        
        # Set default FPS for analysis
        self.fps = 1  # Default to 1 FPS for analysis
        
    def analyze_segment(self, video_path: str, start_time: float, end_time: float) -> Dict:
        """Perform comprehensive analysis of a video segment."""
        video = VideoFileClip(video_path).subclip(start_time, end_time)
        segment_duration = end_time - start_time
        
        # Extract frames for visual analysis
        frames = self._extract_frames(video)
        
        # Analyze visual elements with more detail
        visual_analysis = self._analyze_visual_elements(frames)
        
        # Extract and analyze audio with more detail
        audio_analysis = self._analyze_audio(video)
        
        # Analyze speech and dialogue with more detail
        speech_analysis = self._analyze_speech(video)
        
        # Find optimal commentary points
        commentary_points = self._find_commentary_points(
            speech_analysis['speech_segments'],
            audio_analysis['volume_levels'],
            visual_analysis['scene_changes'],
            segment_duration,
            start_time
        )
        
        # Add segment-specific characteristics
        segment_characteristics = {
            'dominant_visual_elements': self._identify_dominant_elements(visual_analysis),
            'audio_patterns': self._identify_audio_patterns(audio_analysis),
            'speech_patterns': self._identify_speech_patterns(speech_analysis),
            'scene_transitions': self._analyze_transitions(visual_analysis, audio_analysis),
            'emotional_arc': self._analyze_emotional_arc(speech_analysis, audio_analysis)
        }
        
        return {
            'visual_analysis': visual_analysis,
            'audio_analysis': audio_analysis,
            'speech_analysis': speech_analysis,
            'commentary_points': commentary_points,
            'segment_characteristics': segment_characteristics
        }
        
    def _extract_frames(self, video: VideoFileClip, fps: int = 1) -> List[np.ndarray]:
        """Extract frames from video at specified fps."""
        frames = []
        for t in np.arange(0, video.duration, 1/fps):
            frame = video.get_frame(t)
            frames.append(frame)
        return frames
        
    def _analyze_visual_elements(self, frames: List[np.ndarray]) -> Dict:
        """Analyze visual elements in the frames."""
        analysis = {
            'color_palette': [],
            'brightness_levels': [],
            'scene_changes': [],
            'composition_analysis': []
        }
        
        prev_frame = None
        for i, frame in enumerate(frames):
            # Analyze color palette
            colors = self._extract_dominant_colors(frame)
            analysis['color_palette'].append(colors)
            
            # Analyze brightness
            brightness = np.mean(frame)
            analysis['brightness_levels'].append(brightness)
            
            # Detect scene changes
            if prev_frame is not None:
                diff = np.mean(np.abs(frame - prev_frame))
                if diff > 50:  # Threshold for scene change
                    analysis['scene_changes'].append(i)
            
            # Analyze composition (rule of thirds, etc.)
            composition = self._analyze_composition(frame)
            analysis['composition_analysis'].append(composition)
            
            prev_frame = frame
            
        return analysis
        
    def _extract_dominant_colors(self, frame: np.ndarray, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from a frame."""
        pixels = frame.reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        return [(int(r), int(g), int(b)) for r, g, b in colors]
        
    def _analyze_composition(self, frame: np.ndarray) -> Dict:
        """Analyze frame composition."""
        height, width = frame.shape[:2]
        thirds_h = height // 3
        thirds_w = width // 3
        
        # Analyze focus points using edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Check rule of thirds points
        roi_points = [
            (thirds_w, thirds_h),
            (thirds_w * 2, thirds_h),
            (thirds_w, thirds_h * 2),
            (thirds_w * 2, thirds_h * 2)
        ]
        
        focus_scores = []
        for x, y in roi_points:
            roi = edges[y-20:y+20, x-20:x+20]
            score = np.sum(roi) / 255
            focus_scores.append(score)
            
        return {
            'rule_of_thirds_scores': focus_scores,
            'symmetry_score': self._calculate_symmetry(frame)
        }
        
    def _calculate_symmetry(self, frame: np.ndarray) -> float:
        """Calculate symmetry score of a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        mid = width // 2
        left = gray[:, :mid]
        right = gray[:, mid:width]
        right_flipped = cv2.flip(right, 1)
        
        # Adjust sizes if needed
        min_width = min(left.shape[1], right_flipped.shape[1])
        symmetry_score = np.mean(np.abs(left[:, :min_width] - right_flipped[:, :min_width]))
        return 1 - (symmetry_score / 255)  # Normalize to 0-1
        
    def _analyze_audio(self, video: VideoFileClip) -> Dict:
        """Analyze audio elements of the video."""
        # Extract audio
        audio = video.audio
        
        # Save temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
            
        # Load audio with librosa
        y, sr = librosa.load(temp_audio.name)
        
        # Clean up temp file
        os.unlink(temp_audio.name)
        
        # Analyze various audio features
        analysis = {
            'volume_levels': librosa.feature.rms(y=y)[0],
            'pitch_features': {
                'f0': librosa.yin(y, fmin=librosa.note_to_hz('C2'), 
                                fmax=librosa.note_to_hz('C7')),
                'chroma': np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            },
            'tempo': librosa.beat.tempo(y=y)[0],
            'spectral_features': {
                'contrast': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1),
                'centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                'bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            }
        }
        
        return analysis
        
    def _analyze_speech(self, video: VideoFileClip) -> Dict:
        """Analyze speech and dialogue in the video."""
        # Extract audio for analysis
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            video.audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
            
        # Load audio with librosa
        y, sr = librosa.load(temp_audio.name)
        
        # Clean up temp file
        os.unlink(temp_audio.name)
        
        # Analyze audio features that might indicate speech
        analysis = {
            'speech_segments': [],
            'audio_features': {
                'rms': librosa.feature.rms(y=y)[0],  # Root mean square energy
                'zero_crossings': librosa.feature.zero_crossing_rate(y=y)[0],  # Zero crossing rate
                'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # Spectral centroid
            }
        }
        
        # Detect potential speech segments based on audio features
        rms = analysis['audio_features']['rms']
        zcr = analysis['audio_features']['zero_crossings']
        
        # Find segments with characteristics of speech
        # Speech typically has higher RMS and moderate zero crossing rate
        speech_threshold = np.mean(rms) + np.std(rms)
        zcr_threshold = np.mean(zcr) + np.std(zcr)
        
        # Create simple speech segments based on audio features
        frame_duration = len(y) / len(rms)
        current_segment = None
        
        for frame_idx in range(len(rms)):
            if rms[frame_idx] > speech_threshold and zcr[frame_idx] < zcr_threshold:
                if current_segment is None:
                    current_segment = {'start': frame_idx * frame_duration, 'end': (frame_idx + 1) * frame_duration}
                else:
                    current_segment['end'] = (frame_idx + 1) * frame_duration
            else:
                if current_segment is not None:
                    # Only keep segments longer than 0.5 seconds
                    if current_segment['end'] - current_segment['start'] > 0.5:
                        analysis['speech_segments'].append(current_segment)
                    current_segment = None
        
        # Add the last segment if it exists
        if current_segment is not None and current_segment['end'] - current_segment['start'] > 0.5:
            analysis['speech_segments'].append(current_segment)
            
        return analysis
        
    def _find_commentary_points(self, 
                              speech_segments: List[Dict],
                              volume_levels: np.ndarray,
                              scene_changes: List[int],
                              segment_duration: float,
                              segment_start: float) -> List[Dict]:
        """Find optimal points for commentary insertion."""
        commentary_points = []
        
        # Convert volume levels to normalized scores
        volume_scores = (volume_levels - np.min(volume_levels)) / (np.max(volume_levels) - np.min(volume_levels))
        
        # Find quiet segments
        quiet_segments = []
        threshold = 0.3  # Adjust as needed
        min_quiet_duration = int(self.fps * 2)  # Minimum 2 seconds of quiet
        quiet = False
        start = 0
        
        for i, score in enumerate(volume_scores):
            if score < threshold and not quiet:
                quiet = True
                start = i
            elif score >= threshold and quiet:
                quiet = False
                if i - start >= min_quiet_duration:  # Only keep segments long enough
                    quiet_segments.append((start, i))
                
        # If no quiet segments found, create some based on scene changes
        if not quiet_segments and scene_changes:
            for sc in scene_changes:
                quiet_segments.append((sc, sc + min_quiet_duration))
        
        # Score each quiet segment
        for start, end in quiet_segments:
            duration = end - start
            
            # Check if segment is near a scene change
            near_scene_change = any(abs(sc - start) < 30 for sc in scene_changes)  # Within 30 frames
            
            # Convert frame indices to actual timestamps
            relative_start = start / len(volume_scores) * segment_duration
            relative_end = min((end / len(volume_scores) * segment_duration), segment_duration)
            actual_start = segment_start + relative_start
            actual_end = segment_start + relative_end
            
            # Ensure minimum commentary duration
            if actual_end - actual_start < 2.0:  # Minimum 2 seconds
                actual_end = actual_start + 2.0
            
            # Score the segment
            score = {
                'start_time': actual_start,
                'end_time': min(actual_end, segment_start + segment_duration),  # Don't exceed segment
                'duration': min(actual_end, segment_start + segment_duration) - actual_start,
                'volume_level': float(np.mean(volume_scores[start:end])),
                'near_scene_change': near_scene_change,
                'priority': 1.0 if near_scene_change else 0.5  # Prioritize scene changes
            }
            
            commentary_points.append(score)
            
        # Sort by priority and filter overlapping segments
        commentary_points.sort(key=lambda x: (-x['priority'], x['start_time']))
        filtered_points = []
        
        for point in commentary_points:
            # Check if this point overlaps with any already selected points
            overlaps = False
            for selected in filtered_points:
                if (point['start_time'] < selected['end_time'] + 1.0 and  # Add 1 second buffer
                    point['end_time'] > selected['start_time'] - 1.0):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_points.append(point)
                
        # Take at most 2 points per segment to avoid over-commenting
        return filtered_points[:2]

    def _identify_dominant_elements(self, visual_analysis: Dict) -> Dict:
        """Identify the most prominent visual elements in the segment."""
        dominant = {
            'colors': [],
            'composition': [],
            'movement': []
        }
        
        # Analyze color dominance
        color_counts = {}
        for colors in visual_analysis['color_palette']:
            for color in colors:
                color_counts[color] = color_counts.get(color, 0) + 1
        dominant['colors'] = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Analyze composition patterns
        comp_counts = {}
        for comp in visual_analysis['composition_analysis']:
            # Extract composition type from the dictionary
            comp_type = None
            if 'rule_of_thirds_scores' in comp and max(comp['rule_of_thirds_scores']) > 0.7:
                comp_type = 'rule_of_thirds'
            elif 'symmetry_score' in comp and comp['symmetry_score'] > 0.8:
                comp_type = 'symmetry'
            
            if comp_type:
                comp_counts[comp_type] = comp_counts.get(comp_type, 0) + 1
        
        dominant['composition'] = sorted(comp_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        
        return dominant

    def _identify_audio_patterns(self, audio_analysis: Dict) -> Dict:
        """Identify patterns in the audio elements."""
        patterns = {
            'volume_changes': [],
            'pitch_movements': [],
            'rhythmic_elements': []
        }
        
        # Analyze volume changes
        volume = audio_analysis['volume_levels']
        changes = np.diff(volume)
        patterns['volume_changes'] = {
            'max_increase': float(np.max(changes)),
            'max_decrease': float(np.min(changes)),
            'average_change': float(np.mean(np.abs(changes)))
        }
        
        # Analyze pitch movements
        if 'pitch_features' in audio_analysis:
            pitch = audio_analysis['pitch_features']['f0']
            patterns['pitch_movements'] = {
                'range': float(np.max(pitch) - np.min(pitch)),
                'average': float(np.mean(pitch)),
                'variability': float(np.std(pitch))
            }
        
        # Analyze rhythmic elements
        if 'tempo' in audio_analysis:
            patterns['rhythmic_elements'] = {
                'tempo': float(audio_analysis['tempo']),
                'consistency': float(1 - np.std(audio_analysis['volume_levels']))
            }
        
        return patterns

    def _identify_speech_patterns(self, speech_analysis: Dict) -> Dict:
        """Identify patterns in speech and dialogue."""
        patterns = {
            'dialogue_density': 0,
            'emotional_patterns': [],
            'interaction_patterns': []
        }
        
        # Calculate dialogue density
        segments = speech_analysis['speech_segments']
        total_duration = sum(seg['end'] - seg['start'] for seg in segments)
        patterns['dialogue_density'] = total_duration / len(segments) if segments else 0
        
        # Analyze emotional patterns
        if 'audio_features' in speech_analysis:
            emotions = [f.split(': ')[1] for f in speech_analysis['audio_features'] if 'emotion' in f]
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            patterns['emotional_patterns'] = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        
        return patterns

    def _analyze_transitions(self, visual_analysis: Dict, audio_analysis: Dict) -> Dict:
        """Analyze how the scene transitions between different elements."""
        transitions = {
            'visual_changes': [],
            'audio_changes': [],
            'combined_impact': []
        }
        
        # Analyze visual transitions
        scene_changes = visual_analysis['scene_changes']
        if len(scene_changes) > 1:
            transitions['visual_changes'] = {
                'frequency': len(scene_changes),
                'average_duration': float(np.mean(np.diff(scene_changes))),
                'pattern': 'regular' if np.std(np.diff(scene_changes)) < 5 else 'irregular'
            }
        
        # Analyze audio transitions
        volume = audio_analysis['volume_levels']
        transitions['audio_changes'] = {
            'dynamic_range': float(np.max(volume) - np.min(volume)),
            'transition_smoothness': float(1 - np.std(np.diff(volume)))
        }
        
        return transitions

    def _analyze_emotional_arc(self, speech_analysis: Dict, audio_analysis: Dict) -> Dict:
        """Analyze the emotional progression throughout the segment."""
        arc = {
            'start_emotion': None,
            'end_emotion': None,
            'progression': [],
            'climax_points': []
        }
        
        # Analyze emotional progression in speech
        if 'audio_features' in speech_analysis:
            emotions = [f.split(': ')[1] for f in speech_analysis['audio_features'] if 'emotion' in f]
            if emotions:
                arc['start_emotion'] = emotions[0]
                arc['end_emotion'] = emotions[-1]
                arc['progression'] = emotions
        
        # Identify emotional climax points
        if 'volume_levels' in audio_analysis:
            volume = audio_analysis['volume_levels']
            peaks = np.where(volume > np.mean(volume) + np.std(volume))[0]
            arc['climax_points'] = [float(p) for p in peaks]
        
        return arc 