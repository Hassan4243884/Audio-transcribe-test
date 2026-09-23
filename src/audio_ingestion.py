import os
from pathlib import Path
from typing import Union

class AudioIngestionService:
    """
    Accepts and validates audio files across various formats (WAV, MP3, M4A, etc.).
    """
    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}

    def accept_audio(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path)
        
        # 1. Check file existence
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # 2. Validate file extension
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: '{path.suffix}'. Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

        # 3. Validate non-empty file
        if path.stat().st_size == 0:
            raise ValueError("Provided audio file is empty (0 bytes).")

        return path.resolve()

if __name__ == "__main__":
    # Example usage:
    print("AudioIngestionService initialized.")
    # service = AudioIngestionService()
    # valid_audio_path = service.accept_audio("sample_audio.mp3")
    # print(f"Accepted audio file at: {valid_audio_path}")
