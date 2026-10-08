<p align="center">
  <img src="https://raw.githubusercontent.com/coccinella-labs/benchmark/main/.github/assets/thumbnail.png" alt="benchmark" width="100%">
</p>

Compares OpenAI Whisper against Meta's Wav2Vec2 (`facebook/wav2vec2-base-960h`) for transcription quality.

**The comparison is not automated yet.** WER and CER are implemented in `harpertoken/evaluate.py` and unit-tested, but nothing in `main.py` or the training path calls them, so no script here runs both models and reports a score side by side. `main.py` trains a model; that is all it does. The examples below show how to load each model and get output, not how to compare them.

Audio is captured live from the microphone via `sounddevice`. With `CI=true`, or when no input device is available, `LiveSpeechDataset` substitutes one second of silence, so a run can complete without hardware.

## Usage

```bash
pip install -r requirements.txt
python main.py --model_type whisper    # default
python main.py --model_type wav2vec2
```

Publishing a fine-tuned model is opt-in and needs an explicit target. No repository name is hardcoded, because the previous default named a repo that no longer exists:

```bash
FT_UPLOAD=true FT_REPO_ID=harpertoken/talk HF_TOKEN=... python main.py --model_type whisper
```

Without `FT_UPLOAD=true` the run trains and skips the upload. With the flag set but `FT_REPO_ID` or `HF_TOKEN` missing, it fails loudly rather than appearing to publish.

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

## License

Apache 2.0. See LICENSE file.
