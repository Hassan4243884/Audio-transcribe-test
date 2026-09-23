# Audio Transcription Pipeline

A modular, lightweight audio transcription pipeline in Python built on [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper). It supports format validation, full-text speech-to-text, and timestamped phrase segmentation.

---

## Architecture & Modules

The project is structured under [`src/`](src/) with distinct, decoupled components:

- **[`AudioIngestionService`](src/audio_ingestion.py)**: Validates input audio files (checks existence, non-zero file size, and supported extensions: `.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`, `.webm`).
- **[`SpeechToTextService`](src/transcription.py)**: Transcribes speech into unified, complete text output.
- **[`TimestampedTranscriptionService`](src/timestamped_transcription.py)**: Produces segment-level timestamps (`start`, `end`), text, and confidence log-probabilities using the [`TimestampedSegment`](src/timestamped_transcription.py) schema.
- **[`main.py`](main.py)**: End-to-end CLI orchestrating ingestion, plain transcription, and timestamped segmentation.

---

## Key Design Decisions

1. **`faster-whisper` (CTranslate2) Engine**
   - **Decision**: Used `faster-whisper` rather than OpenAI's standard PyTorch Whisper implementation.
   - **Rationale**: CTranslate2 provides up to 4x faster execution and lower memory usage, running efficiently on CPU via `int8` quantization without requiring an external GPU.

2. **Decoupled Ingestion & Validation Layer**
   - **Decision**: Separated file inspection and format validation (`AudioIngestionService`) from transcription logic.
   - **Rationale**: Fails early on corrupt or empty files before loading neural network weights or launching model inference.

3. **Dependency Injection & Composition**
   - **Decision**: Injected `AudioIngestionService` into transcription classes rather than relying on global state or inheritance.
   - **Rationale**: Improves testability (easy to mock ingestion) and allows custom validation strategies to be swapped in.

4. **Structured Dataclass Output for Segments**
   - **Decision**: Encapsulated segment results into a typed [`TimestampedSegment`](src/timestamped_transcription.py) dataclass.
   - **Rationale**: Provides clear contract guarantees (`id`, `start`, `end`, `text`, `avg_logprob`), consistent rounding, and easy JSON/dict serialization.

---

## Quickstart

### 1. Installation

Activate your virtual environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Run Pipeline

Run on the default sample audio (`audio_files/harvard.wav`):

```bash
python main.py
```

Or provide a custom audio file:

```bash
python main.py path/to/your_audio.wav
```
