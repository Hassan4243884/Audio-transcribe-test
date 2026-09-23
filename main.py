import sys
from pathlib import Path

from src.audio_ingestion import AudioIngestionService
from src.transcription import SpeechToTextService
from src.timestamped_transcription import TimestampedTranscriptionService


def main():
    # Use provided path or default to sample test audio
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "audio_files/harvard.wav"

    print("=" * 60)
    print("AUDIO TRANSCRIPTION PIPELINE")
    print(f"File: {audio_path}")
    print("=" * 60)

    # 1. Ingest and validate audio
    print("\n1. Ingesting audio...")
    ingestion = AudioIngestionService()
    valid_path = ingestion.accept_audio(audio_path)
    print(f"✓ Validated: {valid_path}")

    # 2. Transcribe speech to text
    print("\n2. Transcribing audio...")
    stt = SpeechToTextService(model_size="base")
    text = stt.transcribe(valid_path)
    print(f"✓ Transcription:\n{text}")

    # 3. Transcribe with timestamps per segment
    print("\n3. Transcribing with timestamps...")
    ts_service = TimestampedTranscriptionService(model_size="base")
    segments = ts_service.transcribe_with_timestamps(valid_path)
    print("✓ Segments:")
    for seg in segments:
        print(f"  [{seg['start']:05.2f}s -> {seg['end']:05.2f}s]: {seg['text']}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
