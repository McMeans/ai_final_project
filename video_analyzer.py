# System imports
import os

# Disable GPU/MPS before anything else
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_MPS"] = "0"
os.environ["TORCH_DEVICE"] = "cpu"

# Third-party imports
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import pipeline
import speech_recognition as sr
from typing import Dict, List

# Torch configuration
import torch
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.enabled = False
if hasattr(torch.backends, 'cuda'):
    torch.backends.cuda.enabled = False
torch.set_default_tensor_type('torch.FloatTensor')

class VideoAnalyzer:
    def __init__(self):
        """Initialize the video analyzer with necessary models."""
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        print("Initializing models on CPU...")
        
        try:
            # For initial testing, let's use a simpler model setup
            print("Loading basic models...")
            
            # Initialize a simple image classification pipeline
            self.visual_model = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                use_fast=True
            )
            
            # Initialize a simple audio classification pipeline
            self.audio_classifier = pipeline(
                "audio-classification",
                model="superb/hubert-base-superb-sid",
                use_fast=True
            )
            
            print("All models initialized successfully")
            
        except Exception as e:
            print(f"Error during model initialization: {str(e)}")
            raise
            
    def _analyze_audio(self, video: VideoFileClip) -> Dict:
        """Analyze audio elements of the video."""
        try:
            if video.audio is None:
                return {
                    'volume_profile': {
                        'levels': [],
                        'timestamps': [],
                        'characteristics': {
                            'dynamic_range': 0.0,
                            'average_volume': 0.0,
                            'volume_variance': 0.0,
                            'peak_count': 0,
                            'volume_trend': 'unknown'
                        },
                        'peaks': []
                    }
                }

            # Get raw audio data frame by frame
            try:
                fps = video.audio.fps
                duration = video.audio.duration
                frame_duration = 0.1  # 100ms chunks
                timestamps = []
                volume_levels = []
                
                # Process audio in 100ms chunks
                for t in np.arange(0, duration, frame_duration):
                    try:
                        # Get single frame of audio
                        frame = video.audio.get_frame(t)
                        
                        # Handle different frame formats
                        if isinstance(frame, (np.ndarray, list, tuple)):
                            # Convert frame to numpy array if it isn't already
                            frame_array = np.array(frame, dtype=np.float32)
                            
                            # Handle multi-channel audio
                            if len(frame_array.shape) > 0:
                                # If it's a multi-dimensional array, take the mean across all dimensions
                                volume = float(np.abs(frame_array).mean())
                            else:
                                # If it's a single value, just use it
                                volume = float(np.abs(frame_array))
                                
                            volume_levels.append(volume)
                            timestamps.append(float(t))
                            
                    except Exception as frame_error:
                        print(f"Warning: Failed to process frame at {t}s: {str(frame_error)}")
                        continue
                
                if not volume_levels:
                    raise ValueError("No valid audio frames processed")
                
                # Convert to numpy arrays for calculations
                volume_array = np.array(volume_levels, dtype=np.float32)
                timestamps_array = np.array(timestamps, dtype=np.float32)
                
                # Find peaks (significant volume changes)
                peaks = []
                if len(volume_array) > 2:
                    threshold = np.mean(volume_array) + np.std(volume_array)
                    for i in range(1, len(volume_array)-1):
                        if volume_array[i] > threshold:
                            if volume_array[i] > volume_array[i-1] and volume_array[i] > volume_array[i+1]:
                                peaks.append({
                                    'time': float(timestamps_array[i]),
                                    'intensity': float(volume_array[i])
                                })
                
                # Calculate basic audio characteristics
                audio_characteristics = {
                    'dynamic_range': float(np.max(volume_array) - np.min(volume_array)),
                    'average_volume': float(np.mean(volume_array)),
                    'volume_variance': float(np.std(volume_array)),
                    'peak_count': len(peaks),
                    'volume_trend': 'increasing' if len(volume_array) > 1 and volume_array[-1] > volume_array[0] else 'decreasing'
                }
                
                return {
                    'volume_profile': {
                        'levels': volume_array.tolist(),
                        'timestamps': timestamps_array.tolist(),
                        'characteristics': audio_characteristics,
                        'peaks': peaks
                    }
                }
                
            except Exception as e:
                print(f"Warning: Failed to process audio data: {str(e)}")
                return {
                    'volume_profile': {
                        'levels': [],
                        'timestamps': [],
                        'characteristics': {
                            'dynamic_range': 0.0,
                            'average_volume': 0.0,
                            'volume_variance': 0.0,
                            'peak_count': 0,
                            'volume_trend': 'unknown'
                        },
                        'peaks': []
                    }
                }
                
        except Exception as e:
            print(f"Warning: Audio analysis failed: {str(e)}")
            return {
                'volume_profile': {
                    'levels': [],
                    'timestamps': [],
                    'characteristics': {
                        'dynamic_range': 0.0,
                        'average_volume': 0.0,
                        'volume_variance': 0.0,
                        'peak_count': 0,
                        'volume_trend': 'unknown'
                    },
                    'peaks': []
                }
            }

    def analyze_segment(self, video_path: str, start_time: float, end_time: float) -> Dict:
        """Perform comprehensive analysis of a video segment."""
        try:
            print(f"Analyzing segment from {start_time} to {end_time} seconds")
            
            with VideoFileClip(video_path) as video:
                # Get the segment
                video_segment = video.subclip(start_time, end_time)
                fps = video_segment.fps
                duration = video_segment.duration
                
                # Use 2 frames per second for analysis
                analysis_fps = 2
                frame_interval = 1.0 / analysis_fps
                total_frames = int(duration * analysis_fps)
                
                print(f"Extracting frames for analysis (Sampling at {analysis_fps} FPS from original {fps:.2f} FPS)...")
                print(f"Processing {total_frames} frames...")
                
                frames = []
                frame_times = []
                
                # Extract frames at regular intervals
                for frame_idx in range(total_frames):
                    try:
                        # Calculate exact time for this frame
                        frame_time = frame_idx * frame_interval
                        frame = video_segment.get_frame(frame_time)
                        
                        # Ensure frame is in correct format
                        if isinstance(frame, np.ndarray) and frame.size > 0:
                            frames.append(frame)
                            frame_times.append(start_time + frame_time)
                            
                        # Print progress every 25% of frames
                        if frame_idx % max(1, total_frames // 4) == 0:
                            progress = (frame_idx / total_frames) * 100
                            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")
                            
                    except Exception as e:
                        print(f"Warning: Failed to extract frame at {frame_time:.2f}s: {str(e)}")
                
                if not frames:
                    raise ValueError("No valid frames could be extracted")
                
                print(f"Successfully extracted {len(frames)} frames")
                
                print("Analyzing visual composition...")
                # Analyze visual composition
                visual_analysis = {
                    'composition': self._analyze_composition(frames, frame_times),
                    'color_analysis': self._analyze_colors(frames, frame_times),
                    'lighting': self._analyze_lighting(frames, frame_times),
                    'movement': self._analyze_movement(frames, frame_times)
                }
                
                print("Analyzing audio elements...")
                # Audio analysis
                audio_analysis = self._analyze_audio(video_segment)
                
                print("Analyzing scene dynamics...")
                # Scene dynamics analysis
                try:
                    scene_analysis = {
                        'pacing': self._analyze_pacing(visual_analysis, audio_analysis),
                        'emotional_tone': self._analyze_emotional_tone(visual_analysis, audio_analysis),
                        'key_moments': self._identify_key_moments(visual_analysis, audio_analysis)
                    }
                except Exception as e:
                    print(f"Warning: Scene dynamics analysis failed: {str(e)}")
                    scene_analysis = {
                        'pacing': {'error': 'Analysis failed'},
                        'emotional_tone': {'error': 'Analysis failed'},
                        'key_moments': []
                    }
                
                # Add segment metadata
                segment_metadata = {
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'duration': float(duration),
                    'frames_analyzed': len(frames),
                    'original_fps': float(fps),
                    'analysis_fps': float(analysis_fps),
                    'frame_interval': float(frame_interval)
                }
                
                # Ensure all numerical values are Python native types for JSON serialization
                result = {
                    'segment_info': segment_metadata,
                    'visual_analysis': self._ensure_serializable(visual_analysis),
                    'audio_analysis': self._ensure_serializable(audio_analysis),
                    'scene_analysis': self._ensure_serializable(scene_analysis)
                }
                
                print("Analysis completed successfully")
                return result
            
        except Exception as e:
            print(f"Error during segment analysis: {str(e)}")
            return None

    def _analyze_composition(self, frames: List[np.ndarray], times: List[float]) -> List[Dict]:
        """Analyze visual composition of frames."""
        compositions = []
        for frame, time in zip(frames, times):
            try:
                # Ensure frame is in correct format
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    continue
                    
                # Convert to grayscale for edge detection
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame
                
                # Detect edges
                edges = cv2.Canny(gray, 100, 200)
                
                # Analyze rule of thirds
                h, w = gray.shape
                third_h, third_w = h // 3, w // 3
                thirds_grid = [
                    float(edges[third_h:2*third_h, third_w:2*third_w].mean()),  # Center
                    float(edges[:third_h, :].mean()),  # Top
                    float(edges[-third_h:, :].mean()),  # Bottom
                    float(edges[:, :third_w].mean()),  # Left
                    float(edges[:, -third_w:].mean())   # Right
                ]
                
                # Analyze symmetry
                left_half = gray[:, :w//2]
                right_half = cv2.flip(gray[:, w//2:], 1)
                symmetry_score = float(1 - np.abs(left_half - right_half).mean() / 255)
                
                compositions.append({
                    'time': float(time),
                    'rule_of_thirds_intensity': float(np.mean(thirds_grid)),
                    'symmetry_score': symmetry_score,
                    'edge_density': float(edges.mean() / 255)
                })
            except Exception as e:
                print(f"Warning: Failed to analyze composition for frame at {time}s: {str(e)}")
                continue
        
        return compositions

    def _analyze_colors(self, frames: List[np.ndarray], times: List[float]) -> List[Dict]:
        """Analyze color palette and distribution."""
        color_analysis = []
        for frame, time in zip(frames, times):
            try:
                # Ensure frame is in correct format
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    continue
                    
                # Convert to HSV for better color analysis
                if len(frame.shape) == 3:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                else:
                    continue  # Skip grayscale frames
                
                # Analyze color distribution
                hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                sat_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
                val_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
                
                # Find dominant colors
                pixels = frame.reshape(-1, 3)
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=5, n_init=1)
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_.astype(int)
                
                color_analysis.append({
                    'time': float(time),
                    'dominant_colors': colors.tolist(),
                    'color_stats': {
                        'average_hue': float(np.mean(hsv[:,:,0])),
                        'average_saturation': float(np.mean(hsv[:,:,1])),
                        'average_value': float(np.mean(hsv[:,:,2])),
                        'color_variance': float(np.std(frame.reshape(-1, 3)))
                    }
                })
            except Exception as e:
                print(f"Warning: Failed to analyze colors for frame at {time}s: {str(e)}")
                continue
        
        return color_analysis

    def _analyze_lighting(self, frames: List[np.ndarray], times: List[float]) -> List[Dict]:
        """Analyze lighting conditions and patterns."""
        lighting_analysis = []
        for frame, time in zip(frames, times):
            try:
                # Ensure frame is in correct format
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    continue
                    
                # Convert to grayscale for lighting analysis
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame
                
                # Calculate histogram
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = hist.flatten() / hist.sum()
                
                # Calculate lighting metrics
                brightness = float(np.mean(gray))
                contrast = float(np.std(gray))
                
                # Analyze light direction using gradients
                dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(dx**2 + dy**2)
                gradient_direction = np.arctan2(dy, dx)
                
                lighting_analysis.append({
                    'time': float(time),
                    'brightness': brightness,
                    'contrast': contrast,
                    'histogram_stats': {
                        'shadows': float(np.sum(hist[:85])),
                        'midtones': float(np.sum(hist[85:170])),
                        'highlights': float(np.sum(hist[170:]))
                    },
                    'gradient_stats': {
                        'magnitude': float(np.mean(gradient_magnitude)),
                        'direction': float(np.mean(gradient_direction))
                    }
                })
            except Exception as e:
                print(f"Warning: Failed to analyze lighting for frame at {time}s: {str(e)}")
                continue
        
        return lighting_analysis

    def _analyze_movement(self, frames: List[np.ndarray], times: List[float]) -> Dict:
        """Analyze camera and subject movement."""
        movement_analysis = []
        prev_frame = None
        
        for frame, time in zip(frames, times):
            if prev_frame is not None:
                # Calculate optical flow
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Analyze flow patterns
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                movement_analysis.append({
                    'time': time,
                    'movement_intensity': float(np.mean(magnitude)),
                    'movement_direction': float(np.mean(angle)),
                    'movement_variance': float(np.std(magnitude)),
                    'movement_type': self._classify_movement(magnitude, angle)
                })
            
            prev_frame = frame
        
        return movement_analysis

    def _analyze_pacing(self, visual_analysis: Dict, audio_analysis: Dict) -> Dict:
        """Analyze the pacing of the scene."""
        try:
            # Extract visual changes safely
            visual_changes = []
            if 'composition' in visual_analysis:
                compositions = visual_analysis['composition']
                if compositions and isinstance(compositions, list):
                    visual_changes = [comp.get('edge_density', 0.0) for comp in compositions]
            
            # Extract audio changes safely
            audio_changes = []
            if 'volume_profile' in audio_analysis:
                profile = audio_analysis['volume_profile']
                if isinstance(profile, dict) and 'levels' in profile:
                    audio_changes = profile['levels']
            
            # Ensure we have some data to analyze
            if not visual_changes and not audio_changes:
                return {
                    'visual_pace': {'average': 0.0, 'variance': 0.0},
                    'audio_pace': {'average': 0.0, 'variance': 0.0}
                }
            
            # Calculate pacing metrics
            result = {
                'visual_pace': {
                    'average': float(np.mean(visual_changes)) if visual_changes else 0.0,
                    'variance': float(np.std(visual_changes)) if visual_changes else 0.0
                },
                'audio_pace': {
                    'average': float(np.mean(audio_changes)) if audio_changes else 0.0,
                    'variance': float(np.std(audio_changes)) if audio_changes else 0.0
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Warning: Pacing analysis failed: {str(e)}")
            return {
                'visual_pace': {'average': 0.0, 'variance': 0.0},
                'audio_pace': {'average': 0.0, 'variance': 0.0}
            }

    def _analyze_emotional_tone(self, visual_analysis: Dict, audio_analysis: Dict) -> Dict:
        """Analyze the emotional tone of the scene."""
        try:
            # Initialize default values
            color_temps = []
            light_intensity = []
            
            # Safely extract color information
            if 'color_analysis' in visual_analysis:
                colors = visual_analysis['color_analysis']
                if isinstance(colors, list):
                    for color_data in colors:
                        if isinstance(color_data, dict) and 'color_stats' in color_data:
                            stats = color_data['color_stats']
                            r = stats.get('average_hue', 0)
                            g = stats.get('average_saturation', 0)
                            b = stats.get('average_value', 0)
                            if all(isinstance(x, (int, float)) for x in [r, g, b]):
                                temp = (r * 2 + b) / max(g * 3, 1)  # Avoid division by zero
                                color_temps.append(float(temp))
            
            # Safely extract lighting information
            if 'lighting' in visual_analysis:
                lighting = visual_analysis['lighting']
                if isinstance(lighting, list):
                    light_intensity = [light.get('brightness', 0.0) for light in lighting if isinstance(light, dict)]
            
            # Calculate metrics with safe fallbacks
            result = {
                'color_temperature': {
                    'average': float(np.mean(color_temps)) if color_temps else 0.0,
                    'variance': float(np.std(color_temps)) if color_temps else 0.0
                },
                'lighting_mood': {
                    'average_intensity': float(np.mean(light_intensity)) if light_intensity else 0.0,
                    'contrast_level': float(np.std(light_intensity)) if light_intensity else 0.0
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Warning: Emotional tone analysis failed: {str(e)}")
            return {
                'color_temperature': {'average': 0.0, 'variance': 0.0},
                'lighting_mood': {'average_intensity': 0.0, 'contrast_level': 0.0}
            }

    def _identify_key_moments(self, visual_analysis: Dict, audio_analysis: Dict) -> List[Dict]:
        """Identify key moments in the scene."""
        key_moments = []
        
        # Look for significant visual changes
        for i, comp in enumerate(visual_analysis['composition']):
            if i > 0:
                edge_change = abs(comp['edge_density'] - visual_analysis['composition'][i-1]['edge_density'])
                if edge_change > 0.2:  # Significant composition change
                    key_moments.append({
                        'time': comp['time'],
                        'type': 'visual_change',
                        'intensity': float(edge_change)
                    })
        
        # Add audio key moments
        if 'peaks' in audio_analysis['volume_profile']:
            for peak in audio_analysis['volume_profile']['peaks']:
                key_moments.append({
                    'time': peak['time'],
                    'type': 'audio_peak',
                    'intensity': float(peak['intensity'])
                })
        
        # Sort by time
        key_moments.sort(key=lambda x: x['time'])
        return key_moments

    def _classify_movement(self, magnitude: np.ndarray, angle: np.ndarray) -> str:
        """Classify the type of movement based on optical flow."""
        avg_magnitude = np.mean(magnitude)
        angle_hist = np.histogram(angle, bins=8)[0]
        
        if avg_magnitude < 0.1:
            return "static"
        elif np.std(angle_hist) < np.mean(angle_hist) * 0.5:
            return "uniform_motion"
        elif np.max(angle_hist) > np.sum(angle_hist) * 0.4:
            return "directional_motion"
        else:
            return "complex_motion"

    def _ensure_serializable(self, obj):
        """Ensure all values in a nested structure are JSON serializable."""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8,
                          np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._ensure_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._ensure_serializable(item) for item in obj)
        return obj 