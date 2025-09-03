import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import csv
from datetime import datetime
import os

@dataclass
class Segment:
    start: float  # start time in seconds
    end: float    # end time in seconds
    text: str = ""  # optional transcript

def load_audio(mp3_path: str, target_sr: int = 16000):
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

def extract_segment_audio(waveform: torch.Tensor, sample_rate: int, 
                         start_time: float, end_time: float):
    """Extract audio segment from waveform"""
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Ensure we don't go beyond audio bounds
    start_sample = max(0, start_sample)
    end_sample = min(waveform.shape[1], end_sample)
    
    return waveform[:, start_sample:end_sample]

def calculate_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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
        print(f"Error calculating metrics: {e}")
        return {
            "silhouette_score": -1.0,
            "calinski_harabasz_score": 0.0,
            "davies_bouldin_score": float('inf')
        }

def calculate_composite_score(metrics: Dict[str, float]) -> float:
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

def try_clustering_methods(embeddings: np.ndarray, n_clusters: int) -> Dict[str, Any]:
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
        agg_cosine_metrics = calculate_clustering_metrics(embeddings, agg_cosine_labels)
        
        methods["agglomerative_cosine"] = {
            "labels": agg_cosine_labels.tolist(),
            "metrics": agg_cosine_metrics,
            "composite_score": calculate_composite_score(agg_cosine_metrics)
        }
    except Exception as e:
        print(f"Agglomerative cosine failed for {n_clusters} clusters: {e}")
    
    # K-means clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings)
        kmeans_metrics = calculate_clustering_metrics(embeddings, kmeans_labels)
        
        methods["kmeans"] = {
            "labels": kmeans_labels.tolist(),
            "metrics": kmeans_metrics,
            "composite_score": calculate_composite_score(kmeans_metrics)
        }
    except Exception as e:
        print(f"K-means failed for {n_clusters} clusters: {e}")
    
    # Agglomerative clustering with euclidean distance
    try:
        agg_euclidean = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'  # ward only works with euclidean
        )
        agg_euclidean_labels = agg_euclidean.fit_predict(embeddings)
        agg_euclidean_metrics = calculate_clustering_metrics(embeddings, agg_euclidean_labels)
        
        methods["agglomerative_euclidean"] = {
            "labels": agg_euclidean_labels.tolist(),
            "metrics": agg_euclidean_metrics,
            "composite_score": calculate_composite_score(agg_euclidean_metrics)
        }
    except Exception as e:
        print(f"Agglomerative euclidean failed for {n_clusters} clusters: {e}")
    
    return methods

def assign_speakers_to_all_segments(all_segments_with_embeddings: List, 
                                   labels: List[int],
                                   clustering_embeddings: np.ndarray) -> List[int]:
    """
    Assign speakers to all segments based on clustering centroids.
    Uses cosine similarity to find the closest cluster centroid for each segment.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
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

def process_clustering_results(all_segments_with_embeddings: List,
                             labels: List[int], embeddings: np.ndarray) -> Dict[str, Any]:
    """Process clustering results into speaker statistics for ALL segments"""
    
    # Assign speakers to all segments based on clustering
    all_labels = assign_speakers_to_all_segments(
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

def identify_speakers_with_optimization(mp3_path: str, segments: List[Segment],
                                      min_speakers: int = 2, max_speakers: int = 8,
                                      min_segment_duration: float = 1.0,
                                      output_dir: str = "speaker_analysis_results") -> Dict[str, Any]:
    """
    Extract speaker embeddings and try different clustering configurations
    
    Args:
        mp3_path: Path to MP3 file
        segments: List of Segment objects with start/end times
        min_speakers: Minimum number of speakers to try
        max_speakers: Maximum number of speakers to try
        min_segment_duration: Minimum segment duration to process (seconds)
        output_dir: Directory to save results
    
    Returns:
        Dictionary with best clustering results and analysis
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load the pre-trained ECAPA-TDNN model
    print("Loading ECAPA-TDNN model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    
    # Load audio
    print(f"Loading audio from {mp3_path}...")
    waveform, sample_rate = load_audio(mp3_path)
    
    # Extract embeddings for ALL segments
    all_embeddings = []
    all_segments_with_embeddings = []
    clustering_embeddings = []
    clustering_segments = []
    
    print(f"Processing {len(segments)} segments...")
    for i, segment in enumerate(segments):
        duration = segment.end - segment.start
        
        # Extract segment audio
        segment_audio = extract_segment_audio(
            waveform, sample_rate, segment.start, segment.end
        )
        
        # Skip if segment is empty
        if segment_audio.shape[1] == 0:
            print(f"Skipping segment {i} (empty audio)")
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
                print(f"Processed segment {i}: {segment.start:.1f}s-{segment.end:.1f}s (used for clustering)")
            else:
                print(f"Processed segment {i}: {segment.start:.1f}s-{segment.end:.1f}s (embedding only)")
    
    if len(clustering_embeddings) == 0:
        return {"error": "No segments meet minimum duration for clustering"}
    
    # Convert to numpy array for clustering
    clustering_embeddings = np.array(clustering_embeddings)
    
    # Adjust max_speakers if we have fewer clustering segments
    max_speakers = min(max_speakers, len(clustering_embeddings))
    min_speakers = min(min_speakers, max_speakers)
    
    print(f"Testing clustering with {min_speakers} to {max_speakers} speakers...")
    print(f"Using {len(clustering_segments)} segments for clustering, {len(all_segments_with_embeddings)} total segments")
    
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
        print(f"  Trying {n_clusters} speakers...")
        
        cluster_methods = try_clustering_methods(clustering_embeddings, n_clusters)
        
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
        best_results = process_clustering_results(
            all_segments_with_embeddings, 
            best_config["labels"], clustering_embeddings
        )
        
        all_results["best_clustering"] = {
            "config": best_config,
            "results": best_results
        }
        
        print(f"\nBest clustering: {best_config['n_clusters']} speakers using {best_config['method']}")
        print(f"Composite score: {best_config['composite_score']:.3f}")
        print(f"Silhouette score: {best_config['metrics']['silhouette_score']:.3f}")
    
    # Save detailed results to JSON
    json_path = os.path.join(output_dir, f"speaker_analysis_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary to CSV
    csv_path = os.path.join(output_dir, f"clustering_summary_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        if summary_data:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
    
    # Save segment assignments for best clustering
    if best_config:
        segments_csv_path = os.path.join(output_dir, f"segment_assignments_{timestamp}.csv")
        with open(segments_csv_path, 'w', newline='') as f:
            fieldnames = ["segment_index", "start", "end", "duration", "speaker_id", "text"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(best_results["segments"])
    
    print(f"\nResults saved to:")
    print(f"  Detailed analysis: {json_path}")
    print(f"  Clustering summary: {csv_path}")
    if best_config:
        print(f"  Segment assignments: {segments_csv_path}")
    
    all_results["summary"] = summary_data
    return all_results

# Example usage
if __name__ == "__main__":
    # Define your segments
    segments = [
        Segment(start=0.5, end=3.2, text="Hello, how are you today?"),
        Segment(start=4.1, end=7.8, text="I'm doing great, thanks for asking."),
        Segment(start=8.5, end=12.1, text="That's wonderful to hear."),
        Segment(start=13.0, end=16.5, text="Yes, it's been a good day so far."),
        Segment(start=17.2, end=20.8, text="I hope it continues that way."),
        Segment(start=21.5, end=25.1, text="What are your plans for later?"),
        Segment(start=26.0, end=29.8, text="I'm thinking of going for a walk."),
        Segment(start=30.5, end=34.2, text="That sounds like a great idea."),
    ]
    
    # Process the audio with clustering optimization
    mp3_file = "your_audio.mp3"  # Replace with your MP3 file path
    
    try:
        results = identify_speakers_with_optimization(
            mp3_path=mp3_file,
            segments=segments,
            min_speakers=2,
            max_speakers=6,
            min_segment_duration=1.0,
            output_dir="speaker_analysis_results"
        )
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            best = results["best_clustering"]
            if best:
                print(f"\nBest result: {best['config']['n_clusters']} speakers")
                print(f"Method: {best['config']['method']}")
                print(f"Score: {best['config']['composite_score']:.3f}")
                
                lead_speaker = best["results"]["lead_speaker"]
                print(f"Lead speaker: Speaker {lead_speaker['speaker_id']} "
                      f"({lead_speaker['percentage']:.1f}% of speaking time)")
    
    except Exception as e:
        print(f"Error processing audio: {e}")
