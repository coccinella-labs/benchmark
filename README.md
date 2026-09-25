<p align="center">
  <img src="https://raw.githubusercontent.com/Coccinella-Labs/benchmark/main/.github/assets/thumbnail.png" alt="benchmark" width="100%">
</p>

Compares OpenAI Whisper against Meta's Wav2Vec2 (`facebook/wav2vec2-base-960h`) for transcription quality.

## Usage

```bash
pip install -r requirements.txt
python main.py --model_type whisper    # default
python main.py --model_type wav2vec2
```

## Layout

- `harpertoken/` - training package (`train`, `dataset`, `evaluate`, `model`, `preprocessing`)
- `tests/` - transcription and unit tests (see `docs/TESTING.md`)
- `run_tests.py` - test runner (prefers `./venv/bin/python3`)
- `Dockerfile` - containerized runs

## Stack

PyTorch, Transformers, Librosa, Datasets, scikit-learn.

## Examples

### Whisper transcription

```python
from harpertoken.model import SpeechModel
from harpertoken.dataset import LiveSpeechDataset
from transformers import WhisperProcessor
import torch

# use tiny model for faster testing (or whisper-small for better quality)
model = SpeechModel(model_type="whisper")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

dataset = LiveSpeechDataset()
audio = dataset.record_audio()

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(
        input_features=inputs.input_features,
        language="en",
        task="transcribe",
    )

print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

### Wav2Vec2 features

```python
from harpertoken.model import SpeechModel
from harpertoken.dataset import LiveSpeechDataset
from transformers import Wav2Vec2FeatureExtractor
import torch

model = SpeechModel(model_type="wav2vec2")
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

dataset = LiveSpeechDataset()
audio = dataset.record_audio()

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    features = model(inputs.input_values)

print(features.shape)
```

### Train and test

```bash
python run_tests.py
python -m unittest tests.test_unit
python tests/test_transcription.py --model_type whisper
```
