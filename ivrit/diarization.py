"""
Speaker diarization functionality for ivrit.ai
------------------------------------------------------------------------------------------------
This module provides two speaker diarization implementations:

1. PyAnnote engine: Based on PyAnnote.audio pipeline with modifications derived from 
   WhisperX (https://github.com/m-bain/whisperX), originally licensed under the BSD 2-Clause License.

2. ivrit engine: Proprietary ivrit.ai implementation using SpeechBrain ECAPA-TDNN embeddings 
   with advanced clustering optimization techniques.
"""


import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torchaudio
from pyannote.audio import Pipeline
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

from .types import Segment
from .utils import SAMPLE_RATE, load_audio

logger = logging.getLogger(__name__)


class BaseDiarizationEngine(ABC):
    """Base class for speaker diarization engines."""
    
    @abstractmethod
    def diarize(
        self,
        audio: Union[str, npt.NDArray],
        transcription_segments: List[Segment],
        **kwargs
    ) -> List[Segment]:
        """
        Perform speaker diarization on audio and assign speaker labels to transcription segments.
        
        Args:
            audio: Path to audio file or NumPy array containing audio waveform.
            transcription_segments: List of transcription segments to assign speaker labels to.
            **kwargs: Engine-specific parameters.
            
        Returns:
            List of transcription segments with speaker labels assigned.
        """
        pass


class PyannoteDiarizationEngine(BaseDiarizationEngine):
    """Speaker diarization engine using PyAnnote.audio pipeline."""
    
    DEFAULT_CHECKPOINT = "ivrit-ai/pyannote-speaker-diarization-3.1"
    
    def __init__(self):
        pass
    
    def _match_speaker_to_interval(
        self,
        diarization_df: pd.DataFrame,
        start: float,
        end: float,
        fill_nearest: bool = False,
    ) -> Optional[str]:
        """
        Match the best speaker for a given time interval.
        Note: This function modifies the diarization_df in place.
        
        Args:
            diarization_df: Diarization dataframe with columns ['start', 'end', 'speaker']
            start: Start time of the interval
            end: End time of the interval
            fill_nearest: If True, match speakers even when there's no direct time overlap
            
        Returns:
            The speaker ID with the highest intersection, or None if no match found
        """
        # Calculate intersection and union
        diarization_df["intersection"] = np.minimum(diarization_df["end"], end) - np.maximum(diarization_df["start"], start)
        diarization_df["union"] = np.maximum(diarization_df["end"], end) - np.minimum(diarization_df["start"], start)
        
        # Filter based on fill_nearest flag
        if not fill_nearest:
            tmp_df = diarization_df[diarization_df["intersection"] > 0]
        else:
            tmp_df = diarization_df
        
        speaker = None
        
        if len(tmp_df) > 0:
            # Sum over speakers and get the one with highest intersection
            speaker = tmp_df.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
        
        return speaker

    def _assign_speakers(
        self,
        diarization_df: pd.DataFrame,
        transcription_segments: List[Segment],
        fill_nearest: bool = False,
    ) -> List[Segment]:
        """
        Assign speakers to words and segments in the transcript.

        Args:
            diarization_df: Diarization dataframe with columns ['start', 'end', 'speaker']
            transcription_segments: List of Segment objects to augment with speaker labels
            fill_nearest: If True, assign speakers even when there's no direct time overlap

        Returns:
            Updated transcription_segments with speaker assignments
        """
        for seg in transcription_segments:
            # assign speaker to segment (if any)
            speaker = self._match_speaker_to_interval(diarization_df, start=seg.start, end=seg.end, fill_nearest=fill_nearest)
            seg.speakers = [speaker]

            # assign speaker to words
            for word in seg.words:
                if word.start:
                    speaker = self._match_speaker_to_interval(diarization_df, start=word.start, end=word.end, fill_nearest=fill_nearest)
                    word.speaker = speaker
                        
        return transcription_segments

    def diarize(
        self,
        audio: Union[str, npt.NDArray],
        transcription_segments: List[Segment],
        *,
        device: Union[str, torch.device] = "cpu",
        checkpoint_path: Optional[Union[str, Path]] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        use_auth_token: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ) -> List[Segment]:
        """
        Perform speaker diarization using PyAnnote.audio pipeline.
        
        Args:
            audio: Path to audio file or NumPy array containing audio waveform.
            transcription_segments: List of transcription segments to assign speaker labels to.
            device: Device to run on ("cpu", "cuda", or torch.device).
            checkpoint_path: Model checkpoint path.
            num_speakers: Exact number of speakers.
            min_speakers: Minimum number of speakers to consider.
            max_speakers: Maximum number of speakers to consider.
            use_auth_token: Authentication token for model download.
            verbose: Whether to enable verbose logging.
            
        Returns:
            List of transcription segments with speaker labels assigned in-place.
        """
        checkpoint_path = checkpoint_path or self.DEFAULT_CHECKPOINT
        if verbose:
            logger.info(f"Diarizing with pyannote, {checkpoint_path=}, {device=}, {num_speakers=}, {min_speakers=}, {max_speakers=}")

        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(audio, str):
            audio = load_audio(audio)

        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": SAMPLE_RATE,
        }
        diarization_pipeline = Pipeline.from_pretrained(checkpoint_path, use_auth_token=use_auth_token).to(device)
        diarization = diarization_pipeline(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarization_df = pd.DataFrame(
            diarization.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarization_df["start"] = diarization_df["segment"].apply(lambda x: x.start)
        diarization_df["end"] = diarization_df["segment"].apply(lambda x: x.end)
        if verbose:
            logger.info("Diarization completed successfully")
        
        diarized_segments = self._assign_speakers(diarization_df, transcription_segments)
        return diarized_segments


class IvritDiarizationEngine(BaseDiarizationEngine):
    """Speaker diarization engine using SpeechBrain ECAPA-TDNN embeddings with clustering."""
    
    SPEECHBRAIN_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
    
    def __init__(self):
        pass
    
    def _load_audio_speechbrain(self, mp3_path: str, target_sr: int = 16000):
        """Load MP3 file and convert to the format expected by SpeechBrain"""
        waveform, sample_rate = torchaudio.load(mp3_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        return waveform, target_sr

    def _extract_segment_audio(self, waveform: torch.Tensor, sample_rate: int, 
                         start_time: float, end_time: float):
        """Extract audio segment from waveform"""
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure we don't go beyond audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)
        
        return waveform[:, start_sample:end_sample]

    def _calculate_clustering_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate various clustering quality metrics"""
        if len(np.unique(labels)) < 2:
            return {
                "silhouette_score": -1.0,
                "calinski_harabasz_score": 0.0,
                "davies_bouldin_score": float('inf')
            }

        try:
            silhouette = silhouette_score(embeddings, labels, metric='cosine')
            calinski_harabasz = calinski_harabasz_score(embeddings, labels)
            davies_bouldin = davies_bouldin_score(embeddings, labels)
            
            return {
                "silhouette_score": float(silhouette),
                "calinski_harabasz_score": float(calinski_harabasz),
                "davies_bouldin_score": float(davies_bouldin)
            }
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return {
                "silhouette_score": -1.0,
                "calinski_harabasz_score": 0.0,
                "davies_bouldin_score": float('inf')
            }

    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate a composite score for clustering quality
        Higher is better (normalized to 0-1 range)
        """
        # Normalize silhouette score (range -1 to 1) to 0-1
        silhouette_normalized = (metrics["silhouette_score"] + 1) / 2
        
        # Davies-Bouldin: lower is better, so we take 1/(1+score)
        davies_bouldin_normalized = 1 / (1 + metrics["davies_bouldin_score"])
        
        # Calinski-Harabasz: higher is better, normalize by taking sigmoid
        calinski_normalized = 1 / (1 + np.exp(-metrics["calinski_harabasz_score"] / 1000))
        
        # Weighted combination (silhouette is most important for speaker ID)
        composite = (0.6 * silhouette_normalized + 
                    0.2 * davies_bouldin_normalized + 
                    0.2 * calinski_normalized)
        
        return float(composite)

    def _try_clustering_methods(self, embeddings: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Try different clustering methods for given number of clusters"""
        methods = {}
        
        # Agglomerative clustering with cosine distance
        try:
            agg_cosine = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='average',
                metric='cosine'
            )
            agg_cosine_labels = agg_cosine.fit_predict(embeddings)
            agg_cosine_metrics = self._calculate_clustering_metrics(embeddings, agg_cosine_labels)
            
            methods["agglomerative_cosine"] = {
                "labels": agg_cosine_labels.tolist(),
                "metrics": agg_cosine_metrics,
                "composite_score": self._calculate_composite_score(agg_cosine_metrics)
            }
        except Exception as e:
            logger.warning(f"Agglomerative cosine failed for {n_clusters} clusters: {e}")
        
        # K-means clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(embeddings)
            kmeans_metrics = self._calculate_clustering_metrics(embeddings, kmeans_labels)
            
            methods["kmeans"] = {
                "labels": kmeans_labels.tolist(),
                "metrics": kmeans_metrics,
                "composite_score": self._calculate_composite_score(kmeans_metrics)
            }
        except Exception as e:
            logger.warning(f"K-means failed for {n_clusters} clusters: {e}")
        
        # Agglomerative clustering with euclidean distance
        try:
            agg_euclidean = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'  # ward only works with euclidean
            )
            agg_euclidean_labels = agg_euclidean.fit_predict(embeddings)
            agg_euclidean_metrics = self._calculate_clustering_metrics(embeddings, agg_euclidean_labels)
            
            methods["agglomerative_euclidean"] = {
                "labels": agg_euclidean_labels.tolist(),
                "metrics": agg_euclidean_metrics,
                "composite_score": self._calculate_composite_score(agg_euclidean_metrics)
            }
        except Exception as e:
            logger.warning(f"Agglomerative euclidean failed for {n_clusters} clusters: {e}")
        
        return methods

    def _assign_speakers_to_all_segments(self, all_segments_with_embeddings: List, 
                                   labels: List[int],
                                   clustering_embeddings: np.ndarray) -> List[int]:
        """
        Assign speakers to all segments based on clustering centroids.
        Uses cosine similarity to find the closest cluster centroid for each segment.
        """
        # Calculate cluster centroids from clustering segments
        unique_labels = np.unique(labels)
        centroids = {}
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_embeddings = clustering_embeddings[cluster_indices]
            centroids[label] = np.mean(cluster_embeddings, axis=0)
        
        # Assign all segments to closest centroid
        all_labels = []
        
        for seg_idx, segment, embedding in all_segments_with_embeddings:
            if embedding is not None:
                # Find closest centroid using cosine similarity
                similarities = {}
                for label, centroid in centroids.items():
                    similarity = cosine_similarity([embedding], [centroid])[0][0]
                    similarities[label] = similarity
                
                # Assign to most similar cluster
                best_label = max(similarities.keys(), key=lambda x: similarities[x])
                all_labels.append(best_label)
            else:
                # No embedding available, assign -1
                all_labels.append(-1)
        
        return all_labels

    def _process_clustering_results(self, all_segments_with_embeddings: List,
                             labels: List[int], embeddings: np.ndarray) -> Dict[str, Any]:
        """Process clustering results into speaker statistics for ALL segments"""
        
        # Assign speakers to all segments based on clustering
        all_labels = self._assign_speakers_to_all_segments(
            all_segments_with_embeddings, labels, embeddings
        )
        
        results = {
            "segments": [],
            "speaker_stats": {},
            "lead_speaker": None,
            "num_speakers": len(np.unique(labels))
        }
        
        speaker_durations = {}
        speaker_segment_counts = {}
        
        # Process all segments with their assigned speakers
        for (seg_idx, segment, _), speaker_id in zip(all_segments_with_embeddings, all_labels):
            duration = segment.end - segment.start
            
            results["segments"].append({
                "segment_index": seg_idx,
                "start": segment.start,
                "end": segment.end,
                "duration": duration,
                "speaker_id": int(speaker_id),
                "text": segment.text
            })
            
            # Track speaker statistics
            if speaker_id not in speaker_durations:
                speaker_durations[speaker_id] = 0
                speaker_segment_counts[speaker_id] = 0
            
            speaker_durations[speaker_id] += duration
            speaker_segment_counts[speaker_id] += 1
        
        # Calculate speaker statistics
        for speaker_id in speaker_durations:
            results["speaker_stats"][f"speaker_{speaker_id}"] = {
                "total_duration": speaker_durations[speaker_id],
                "segment_count": speaker_segment_counts[speaker_id],
                "avg_segment_duration": speaker_durations[speaker_id] / speaker_segment_counts[speaker_id]
            }
        
        # Identify lead speaker (most speaking time)
        if speaker_durations:
            lead_speaker_id = max(speaker_durations.keys(), key=lambda x: speaker_durations[x])
            results["lead_speaker"] = {
                "speaker_id": int(lead_speaker_id),
                "total_duration": speaker_durations[lead_speaker_id],
                "percentage": (speaker_durations[lead_speaker_id] / 
                            sum(speaker_durations.values())) * 100
            }
        
        return results

    def _identify_speakers_with_optimization(self, mp3_path: str, segments: List[Segment],
                                          min_speakers, max_speakers,
                                          min_segment_duration: float = 1.0,
                                          device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
        """
        Extract speaker embeddings and try different clustering configurations
        
        Args:
            mp3_path: Path to MP3 file
            segments: List of Segment objects with start/end times
            min_speakers: Minimum number of speakers to try
            max_speakers: Maximum number of speakers to try
            min_segment_duration: Minimum segment duration to process (seconds)
            device: Device to run on ("cpu", "cuda", or torch.device)
        
        Returns:
            Dictionary with best clustering results and analysis
        """
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError("speechbrain is required for the 'ivrit' diarization engine. Install with: pip install speechbrain")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load the pre-trained ECAPA-TDNN model
        logger.info("Loading ECAPA-TDNN model...")
        classifier = EncoderClassifier.from_hparams(
                source=self.SPEECHBRAIN_MODEL_SOURCE,
                run_opts={'device': str(device)}
        )
        
        # Load audio
        logger.info(f"Loading audio from {mp3_path}")
        waveform, sample_rate = self._load_audio_speechbrain(mp3_path)
        
        # Extract embeddings for ALL segments
        all_embeddings = []
        all_segments_with_embeddings = []
        clustering_embeddings = []
        clustering_segments = []
        
        logger.info(f"Processing {len(segments)} segments...")
        for i, segment in enumerate(segments):
            duration = segment.end - segment.start
            
            # Extract segment audio
            segment_audio = self._extract_segment_audio(
                waveform, sample_rate, segment.start, segment.end
            )
            
            # Skip if segment is empty
            if segment_audio.shape[1] == 0:
                logger.debug(f"Skipping segment {i} (empty audio)")
                all_segments_with_embeddings.append((i, segment, None))
                continue
            
            # Extract embedding for ALL segments
            with torch.no_grad():
                embedding = classifier.encode_batch(segment_audio)
                embedding_np = embedding.squeeze().cpu().numpy()
                all_embeddings.append(embedding_np)
                all_segments_with_embeddings.append((i, segment, embedding_np))
                
                # Only add to clustering if duration meets minimum
                if duration >= min_segment_duration:
                    clustering_embeddings.append(embedding_np)
                    clustering_segments.append((i, segment))
                    logger.debug(f"Processed segment {i}: {segment.start:.1f}s-{segment.end:.1f}s (used for clustering)")
                else:
                    logger.debug(f"Processed segment {i}: {segment.start:.1f}s-{segment.end:.1f}s (embedding only)")
        
        if len(clustering_embeddings) == 0:
            return {"error": "No segments meet minimum duration for clustering"}
        
        # Convert to numpy array for clustering
        clustering_embeddings = np.array(clustering_embeddings)
        
        # Adjust max_speakers if we have fewer clustering segments
        max_speakers = min(max_speakers, len(clustering_embeddings))
        min_speakers = min(min_speakers, max_speakers)
        
        logger.info(f"Testing clustering with {min_speakers} to {max_speakers} speakers...")
        logger.info(f"Using {len(clustering_segments)} segments for clustering, {len(all_segments_with_embeddings)} total segments")
        
        # Try different numbers of clusters
        all_results = {
            "metadata": {
                "mp3_path": mp3_path,
                "total_segments": len(segments),
                "segments_with_embeddings": len(all_segments_with_embeddings),
                "segments_used_for_clustering": len(clustering_segments),
                "min_speakers_tested": min_speakers,
                "max_speakers_tested": max_speakers,
                "min_segment_duration": min_segment_duration,
                "timestamp": timestamp
            },
            "clustering_attempts": {},
            "best_clustering": None,
            "summary": []
        }
        
        best_score = -1
        best_config = None
        summary_data = []
        
        for n_clusters in range(min_speakers, max_speakers + 1):
            logger.debug(f"Trying {n_clusters} speakers...")
            
            cluster_methods = self._try_clustering_methods(clustering_embeddings, n_clusters)
            
            # Store results for this number of clusters
            all_results["clustering_attempts"][str(n_clusters)] = cluster_methods
            
            # Find best method for this number of clusters
            for method_name, method_results in cluster_methods.items():
                score = method_results["composite_score"]
                
                summary_entry = {
                    "n_clusters": n_clusters,
                    "method": method_name,
                    "composite_score": score,
                    "silhouette_score": method_results["metrics"]["silhouette_score"],
                    "calinski_harabasz_score": method_results["metrics"]["calinski_harabasz_score"],
                    "davies_bouldin_score": method_results["metrics"]["davies_bouldin_score"]
                }
                summary_data.append(summary_entry)
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        "n_clusters": n_clusters,
                        "method": method_name,
                        "labels": method_results["labels"],
                        "metrics": method_results["metrics"],
                        "composite_score": score
                    }
        
        # Process best clustering results
        if best_config:
            best_results = self._process_clustering_results(
                all_segments_with_embeddings, 
                best_config["labels"], clustering_embeddings
            )
            
            all_results["best_clustering"] = {
                "config": best_config,
                "results": best_results
            }
            
            logger.info(f"Best clustering: {best_config['n_clusters']} speakers using {best_config['method']}")
            logger.info(f"Composite score: {best_config['composite_score']:.3f}")
            logger.info(f"Silhouette score: {best_config['metrics']['silhouette_score']:.3f}")
        
        
        all_results["summary"] = summary_data
        return all_results

    def diarize(
        self,
        audio: Union[str, npt.NDArray],
        transcription_segments: List[Segment],
        *,
        device: Union[str, torch.device] = "cpu",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        min_segment_duration: float = 1.0,
        verbose: bool = False,
        **kwargs
    ) -> List[Segment]:
        """
        Perform speaker diarization using SpeechBrain ECAPA-TDNN embeddings with clustering.
        
        Args:
            audio: Path to audio file or NumPy array containing audio waveform.
            transcription_segments: List of transcription segments to assign speaker labels to.
            min_speakers: Minimum number of speakers to consider.
            max_speakers: Maximum number of speakers to consider.
            min_segment_duration: Minimum segment duration for clustering.
            verbose: Whether to enable verbose logging.
            
        Returns:
            List of transcription segments with speaker labels assigned in-place.
            
        Raises:
            ValueError: If audio is not a file path.
            ImportError: If speechbrain is not available.
            RuntimeError: If diarization fails.
        """
        if isinstance(audio, str):
            audio_path = audio
        else:
            raise ValueError("ivrit engine currently only supports audio file paths, not numpy arrays")
        
        # Determine min/max speakers for clustering
        clustering_min_speakers = min_speakers or 1
        clustering_max_speakers = max_speakers or 20
        
        # Run the optimization
        results = self._identify_speakers_with_optimization(
            mp3_path=audio_path,
            segments=transcription_segments,
            min_speakers=clustering_min_speakers,
            max_speakers=clustering_max_speakers,
            min_segment_duration=min_segment_duration,
            device=device
        )
        
        if "error" in results:
            raise RuntimeError(f"Diarization failed: {results['error']}")
        
        if not results["best_clustering"]:
            raise RuntimeError("No suitable clustering configuration found")
        
        # Extract the speaker assignments from the best clustering
        best_results = results["best_clustering"]["results"]
        
        # Assign speaker information to segments
        for i, segment in enumerate(transcription_segments):
            speaker_id = best_results["segments"][i]["speaker_id"]
            segment.speakers = [f"SPEAKER_{speaker_id:02d}"]
                
            # Also assign speaker to words if available
            for word in segment.words:
                word.speaker = f"SPEAKER_{speaker_id:02d}"
        
        return transcription_segments


def diarize(
    audio: Union[str, npt.NDArray],
    transcription_segments: List[Segment],
    *,
    engine: str,
    device: Union[str, torch.device] = "cpu",
    checkpoint_path: Optional[Union[str, Path]] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    min_segment_duration: float = 1.0,
    use_auth_token: Optional[str] = None,
    verbose: bool = False,
) -> List[Segment]:
    """
    Perform speaker diarization on audio and assign speaker labels to transcription segments.

    This is the canonical diarization function supporting multiple engines:
    - "pyannote": Uses PyAnnote.audio pipeline for neural speaker diarization
    - "ivrit": Uses SpeechBrain ECAPA-TDNN embeddings with clustering optimization
    
    Args:
        audio: Path to audio file or NumPy array containing audio waveform.
        transcription_segments: List of transcription segments to assign speaker labels to.
        engine: Diarization engine - must be "pyannote" or "ivrit".
        device: Device to run on ("cpu", "cuda", or torch.device).
        checkpoint_path: Model checkpoint path (pyannote engine only).
        num_speakers: Exact number of speakers (pyannote engine only).
        min_speakers: Minimum number of speakers to consider.
        max_speakers: Maximum number of speakers to consider.
        min_segment_duration: Minimum segment duration for clustering (ivrit engine only).
        use_auth_token: Authentication token for model download (pyannote engine only).
        verbose: Whether to enable verbose logging.

    Returns:
        List of transcription segments with speaker labels assigned in-place.
        
    Raises:
        ValueError: If engine is not "pyannote" or "ivrit".
        ImportError: If required dependencies are missing.
        RuntimeError: If diarization fails.

    Examples:
        # Using PyAnnote engine
        diarized_segments = diarize(
            audio="audio.mp3",
            transcription_segments=segments,
            engine="pyannote",
            min_speakers=2,
            max_speakers=4
        )
        
        # Using Ivrit engine  
        diarized_segments = diarize(
            audio="audio.mp3",
            transcription_segments=segments,
            engine="ivrit",
            min_speakers=2,
            max_speakers=6,
            min_segment_duration=1.0
        )
    """
    if engine not in ["pyannote", "ivrit"]:
        raise ValueError(f"Unsupported engine: {engine}. Must be 'pyannote' or 'ivrit'")
    
    if engine == "pyannote":
        engine_instance = PyannoteDiarizationEngine()
        return engine_instance.diarize(
            audio=audio,
            transcription_segments=transcription_segments,
            device=device,
            checkpoint_path=checkpoint_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            use_auth_token=use_auth_token,
            verbose=verbose,
        )
    
    elif engine == "ivrit":
        engine_instance = IvritDiarizationEngine()
        return engine_instance.diarize(
            audio=audio,
            transcription_segments=transcription_segments,
            device=device,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            min_segment_duration=min_segment_duration,
            verbose=verbose,
        )
