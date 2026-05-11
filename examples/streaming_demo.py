import numpy as np
from dataclasses import dataclass
from typing import List, Generator, Optional
import time


@dataclass
class StreamingWord:
    """A word detected during streaming transcription."""
    word: str
    start_time: float
    end_time: float
    confidence: float
    confirmed: bool = False


class WhisperStreamingAnalyzer:
    """
    Prototype for word-level streaming with ivrit-ai's Whisper model.
    
    This is a reference implementation of the recommended approach from
    docs/streaming-analysis.md — Option 3 (VAD + Adaptive Chunking).
    
    For production, replace the stub methods with actual Whisper + VAD calls.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        retry_threshold: float = 0.60,
        chunk_duration: float = 3.0,
        sample_rate: int = 16000,
    ):
        self.confidence_threshold = confidence_threshold
        self.retry_threshold = retry_threshold
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.audio_buffer: List[float] = []
        self.pending_words: List[StreamingWord] = []
        self.confirmed_words: List[StreamingWord] = []
        self.is_speaking = False
        self.total_audio_seconds = 0.0
        
    def _detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Stub: Voice Activity Detection.
        
        In production, use Silero VAD or WebRTC VAD:
            from silero_vad import load_silero_vad, get_speech_timestamps
            model, _ = load_silero_vad()
            timestamps = get_speech_timestamps(audio, model, sampling_rate=16000)
            return len(timestamps) > 0
        """
        # Simple energy-based VAD stub
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return energy > 0.01
    
    def _transcribe_chunk(self, audio: np.ndarray) -> List[StreamingWord]:
        """
        Stub: Transcribe audio chunk with ivrit-ai's Whisper model.
        
        In production, call ivrit-ai's model with word_timestamps=True:
            import whisper
            model = whisper.load_model("ivrit-ai/whisper-large-v3-tuned")
            result = model.transcribe(audio, word_timestamps=True, language="he")
            words = []
            for segment in result["segments"]:
                for word in segment.get("words", []):
                    words.append(StreamingWord(
                        word=word["word"],
                        start_time=word["start"],
                        end_time=word["end"],
                        confidence=word.get("probability", 0.0),
                    ))
            return words
        """
        # Stub returns empty for demonstration
        return []
    
    def _compute_word_confidence(self, word: StreamingWord) -> float:
        """Return normalized confidence score."""
        return max(0.0, min(1.0, word.confidence))
    
    def _retry_pending(self) -> List[StreamingWord]:
        """
        Re-transcribe pending low-confidence words with more context.
        Returns newly confirmed words.
        """
        newly_confirmed = []
        still_pending = []
        
        for word in self.pending_words:
            # In production: extend audio buffer and re-transcribe
            # with additional context, then compare confidence
            
            # Stub: simulate 20% confidence boost with more context
            boosted_confidence = min(1.0, word.confidence + 0.20)
            
            if boosted_confidence >= self.confidence_threshold:
                word.confidence = boosted_confidence
                word.confirmed = True
                newly_confirmed.append(word)
            else:
                still_pending.append(word)
        
        self.pending_words = still_pending
        return newly_confirmed
    
    def ingest_audio(self, audio_chunk: np.ndarray) -> Generator[StreamingWord, None, None]:
        """
        Process incoming audio and yield confirmed words.
        
        Usage:
            analyzer = WhisperStreamingAnalyzer()
            for chunk in audio_stream:
                for word in analyzer.ingest_audio(chunk):
                    print(f"{word.word} ({word.confidence:.2f})")
        """
        # Add to buffer
        self.audio_buffer.extend(audio_chunk.tolist())
        self.total_audio_seconds += len(audio_chunk) / self.sample_rate
        
        # VAD check
        speech_detected = self._detect_speech(audio_chunk)
        
        if speech_detected and not self.is_speaking:
            # Speech started
            self.is_speaking = True
            
        elif not speech_detected and self.is_speaking:
            # Speech ended — process the utterance
            self.is_speaking = False
            
            # Convert buffer to numpy array
            audio_np = np.array(self.audio_buffer, dtype=np.float32)
            
            # Transcribe
            detected_words = self._transcribe_chunk(audio_np)
            
            # Confidence filtering
            for word in detected_words:
                confidence = self._compute_word_confidence(word)
                
                if confidence >= self.confidence_threshold:
                    word.confidence = confidence
                    word.confirmed = True
                    self.confirmed_words.append(word)
                    yield word
                elif confidence >= self.retry_threshold:
                    word.confidence = confidence
                    self.pending_words.append(word)
                else:
                    # Below retry threshold — discard
                    pass
            
            # Retry pending words with accumulated context
            if len(self.audio_buffer) > self.chunk_duration * self.sample_rate * 2:
                for word in self._retry_pending():
                    yield word
            
            # Clear buffer (or keep last 0.5s for context in production)
            self.audio_buffer = []
    
    def get_stats(self) -> dict:
        """Return streaming statistics."""
        return {
            "total_audio_seconds": self.total_audio_seconds,
            "confirmed_words": len(self.confirmed_words),
            "pending_words": len(self.pending_words),
            "average_confidence": (
                np.mean([w.confidence for w in self.confirmed_words])
                if self.confirmed_words else 0.0
            ),
        }


def demo():
    """Demonstrate the streaming analyzer with synthetic audio."""
    analyzer = WhisperStreamingAnalyzer(
        confidence_threshold=0.85,
        retry_threshold=0.60,
        chunk_duration=2.0,
    )
    
    print("=" * 50)
    print("Whisper Streaming Analyzer Demo")
    print("=" * 50)
    
    # Simulate audio stream: 10 seconds of synthetic audio
    # In reality, this would come from a microphone or audio file
    np.random.seed(42)
    
    for i in range(10):
        # Generate 1-second chunk
        chunk = np.random.randn(16000) * 0.02  # Low energy = silence
        
        # Every 3rd chunk, simulate speech (higher energy)
        if i % 3 == 1:
            chunk = np.random.randn(16000) * 0.1
        
        words = list(analyzer.ingest_audio(chunk))
        for word in words:
            print(f"✅ Word: '{word.word}' | conf: {word.confidence:.2f}")
    
    stats = analyzer.get_stats()
    print(f"\n📊 Stats: {stats}")
    print("\nNote: This demo uses synthetic audio.")
    print("For real transcription, integrate ivrit-ai's Whisper model.")


if __name__ == "__main__":
    demo()
