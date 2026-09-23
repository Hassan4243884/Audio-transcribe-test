from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Union
from faster_whisper import WhisperModel

from .audio_ingestion import AudioIngestionService


@dataclass
class TimestampedSegment:
    id: int
    start: float       # Start time in seconds
    end: float         # End time in seconds
    text: str          # Transcribed text in segment
    avg_logprob: float  # Confidence score / avg log-probability


class TimestampedTranscriptionService:
    """
    Transcribes audio and returns timestamped segments per spoken phrase.
    Composes AudioIngestionService to validate and accept audio before transcription.
    """
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        ingestion_service: AudioIngestionService = None
    ):
        self.ingestion_service = ingestion_service or AudioIngestionService()
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_with_timestamps(self, audio_file: Union[str, Path]) -> List[Dict[str, Any]]:
        # 1. Accept and validate audio file
        valid_path = self.ingestion_service.accept_audio(audio_file)

        # 2. Transcribe with segment-level time intervals
        segments, info = self.model.transcribe(str(valid_path), beam_size=5)

        # 3. Format per-segment timestamps
        results: List[Dict[str, Any]] = []
        for idx, seg in enumerate(segments):
            segment_record = TimestampedSegment(
                id=idx,
                start=round(seg.start, 2),
                end=round(seg.end, 2),
                text=seg.text.strip(),
                avg_logprob=round(seg.avg_logprob, 3)
            )
            results.append(asdict(segment_record))

        return results
