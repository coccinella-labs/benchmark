import argparse

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from harpertoken.dataset import LiveSpeechDataset


def test_transcription(model_type="whisper", use_pretrained=False):
    # Load model for testing
    if use_pretrained:
        model_name = "harpertoken/talk"
    elif model_type == "whisper":
        model_name = "openai/whisper-small"
    else:
        model_name = "facebook/wav2vec2-base-960h"

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    # Record new audio
    dataset = LiveSpeechDataset()
    audio = dataset.record_audio()

    # Process audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    # Create attention mask if not present
    if not hasattr(inputs, "attention_mask") or inputs.attention_mask is None:
        inputs.attention_mask = torch.ones(
            inputs.input_features.shape[0],
            inputs.input_features.shape[1],
            dtype=torch.long,
        )

    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(
            input_features=inputs.input_features,
            attention_mask=inputs.attention_mask,
            language="en",
            task="transcribe",
        )

    # Decode transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Transcription: {transcription}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="whisper",
        help="Model type to use (whisper or wav2vec2)",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Use pretrained harpertoken/talk model",
    )
    args = parser.parse_args()

    test_transcription(model_type=args.model_type, use_pretrained=args.use_pretrained)
