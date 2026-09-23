from pathlib import Path
from typing import Union
from faster_whisper import WhisperModel
from .audio_ingestion import AudioIngestionService

class SpeechToTextService:
    """
    Transcribes spoken language into text.
    Composes AudioIngestionService to validate and accept audio before transcription.
    """
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        ingestion_service: AudioIngestionService = None
    ):
        # 1. Dependency injection of the ingestion service from Question 1
        self.ingestion_service = ingestion_service or AudioIngestionService()
        
        # 2. Load open-source Whisper model (CTranslate2 optimized)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_file: Union[str, Path]) -> str:
        # Step 1: Validate and accept the audio file using Question 1's service
        valid_path = self.ingestion_service.accept_audio(audio_file)

        # Step 2: Transcribe spoken language into text
        segments, info = self.model.transcribe(str(valid_path), beam_size=5)

        # Step 3: Combine all spoken segments into full text
        full_text = " ".join(seg.text.strip() for seg in segments)
        return full_text


# Example usage:
# service = SpeechToTextService(model_size="base")
# result = service.transcribe("meeting_recording.mp3")
# print("Transcribed Text:", result)
