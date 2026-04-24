import lightning as L
import torch
import yaml
import argparse
import csv
import sys
from pathlib import Path
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ai_music.train import LightningModel
from ai_music.models import resnet
from ai_music.models.sonics import SpecTTTraAttentionClassifier
from ai_music.data import cross_attention

# Import feature extraction methods from dataset
sys.path.insert(0, str("/home/lennon/AI_music/ISMIR2019-Large-Vocabulary-Chord-Recognition"))
sys.path.insert(0, str("/home/lennon/AI_music/beat_this"))
sys.path.insert(0, "/home/lennon/AI_music")
from beat_this.inference import load_model, LogMelSpect
from feature_extractor import FeatureExtractor
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torchcrepe
import whisper
import warnings
warnings.filterwarnings("ignore")


class InferenceDataset(Dataset):
    """Dataset for inference on a single song directory."""
    def __init__(self, song_dir, data_configs):
        """
        Args:
            song_dir: Path to song directory containing vocals.wav and accompaniment.wav
            data_configs: Data configuration dictionary
        """
        self.song_dir = Path(song_dir)
        self.vocals_path = self.song_dir / 'vocals.wav'
        self.accompaniment_path = self.song_dir / 'accompaniment.wav'
        
        if not self.vocals_path.exists():
            raise FileNotFoundError(f"Vocals file not found: {self.vocals_path}")
        if not self.accompaniment_path.exists():
            raise FileNotFoundError(f"Accompaniment file not found: {self.accompaniment_path}")
        
        self.sr = data_configs["sample_rate"]
        self.duration = data_configs["duration"]
        self.whisper_size = data_configs["whisper_size"]
        self.crepe_size = data_configs["crepe_size"]
        
        # Initialize models
        self.whisper = whisper.load_model(self.whisper_size, device='cuda')
        self.chordnet = FeatureExtractor()
        self.beat_this = load_model('/home/lennon/AI_music/beat_this/final0.ckpt', device='cuda')
        self.bt_spec_extractor = LogMelSpect(
            sample_rate=22050,
            n_fft=1024,
            hop_length=441,
            n_mels=128,
            device='cuda'
        )
        self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        self.mert_model = self.mert_model.to('cuda')
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        
        # Load audio
        v_audio, sr = torchaudio.load(self.vocals_path)
        a_audio, sr = torchaudio.load(self.accompaniment_path)
        
        if v_audio.shape[0] > 1:
            v_audio = v_audio.float().mean(dim=0, keepdim=True)
        if a_audio.shape[0] > 1:
            a_audio = a_audio.float().mean(dim=0, keepdim=True)
        
        if self.sr and (sr != self.sr):
            transform = torchaudio.transforms.Resample(sr, self.sr)
            v_audio = transform(v_audio)
            a_audio = transform(a_audio)
        
        # Extract first clip (duration seconds) or pad if shorter
        self.duration_samples = int(self.duration * self.sr)
        audio_length = min(len(v_audio[0]), len(a_audio[0]))
        
        # Take first clip of specified duration
        v_clip = v_audio[:, :min(self.duration_samples, audio_length)]
        a_clip = a_audio[:, :min(self.duration_samples, audio_length)]
        
        # Pad if shorter than duration
        if v_clip.shape[1] < self.duration_samples:
            padding = self.duration_samples - v_clip.shape[1]
            v_clip = torch.nn.functional.pad(v_clip, (0, padding))
            a_clip = torch.nn.functional.pad(a_clip, (0, padding))
        
        self.v_clip = v_clip
        self.a_clip = a_clip
    
    def __len__(self):
        return 1  # Always return one clip
    
    def __getitem__(self, idx):
        # Extract embeddings from the single clip
        whisper_emb = self._lyrics_emb(self.v_clip)
        crepe_emb = self._pitch_emb(self.v_clip)
        chord_emb = self._chord_emb(self.a_clip)
        beat_emb = self._beat_emb(self.a_clip)
        mert_emb = self._mert_emb(self.a_clip)
        
        # Move embeddings to CPU for dataloader
        embeddings = (whisper_emb.detach().cpu(), crepe_emb.detach().cpu(), chord_emb.detach().cpu(), 
                      beat_emb.detach().cpu(), mert_emb.detach().cpu())
        
        return {"emb": embeddings}
    
    def _pitch_emb(self, clip):
        crepe = torchcrepe.embed(
            clip, self.sr,
            hop_length=int(self.sr/100),
            model="tiny",
            batch_size=512,
            device='cuda',
            pad=True
        ).flatten(start_dim=2)
        return crepe[:, 1:, :]

    def _lyrics_emb(self, clip):
        whisper_embs = []
        for chunk in self._audio_chunks(clip, self.sr*30):
            mel = whisper.log_mel_spectrogram(chunk.squeeze(0))
            with torch.no_grad():
                result = self.whisper.encoder(mel.unsqueeze(0).to('cuda'))
                whisper_embs.append(result)
        return torch.cat(whisper_embs, dim=1)

    def _chord_emb(self, clip):
        chordnet = self.chordnet.extract_features_from_audio(clip.squeeze(0), self.sr)
        return torch.from_numpy(chordnet.astype(np.float32, copy=False)).unsqueeze(0)
    
    def _beat_emb(self, clip):
        if self.sr != 22050:
            resampler = torchaudio.transforms.Resample(self.sr, 22050)
            clip = resampler(clip)

        mono = clip.to('cuda')
        if mono.ndim == 2:
            mono = mono.mean(dim=0) if mono.shape[0] > 1 else mono.squeeze(0)
        elif mono.ndim != 1:
            raise ValueError(f"Expected 1D or 2D audio, got shape {tuple(mono.shape)}")

        spect = self.bt_spec_extractor(mono).unsqueeze(0)
        with torch.inference_mode():
            model_output = self.beat_this(spect)
            beat_embedding = model_output["feat"]
        return beat_embedding
    
    def _mert_emb(self, clip):
        mert_inputs = self.mert_processor(clip.squeeze(0), sampling_rate=24000, return_tensors="pt")
        mert_inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in mert_inputs.items()}
        with torch.no_grad():
            mert_output = self.mert_model(**mert_inputs, output_hidden_states=True)
        mert_all_layer_hidden_states = torch.stack(mert_output.hidden_states).squeeze()
        return mert_all_layer_hidden_states
    
    def _audio_chunks(self, clip, chunk_length):
        n_samples = clip.shape[1]
        n_chunks = n_samples // chunk_length
        trimmed = clip[:, :n_chunks * chunk_length]
        chunks = torch.split(trimmed, chunk_length, dim=1)
        return chunks


def collate_fn(batch):
    """Collate function for inference."""
    embeddings = [item['emb'] for item in batch]
    
    # Stack each embedding type across the batch
    whisper_batch = torch.stack([emb[0] for emb in embeddings])
    crepe_batch = torch.stack([emb[1] for emb in embeddings])
    chord_batch = torch.stack([emb[2] for emb in embeddings])
    beat_this_batch = torch.stack([emb[3] for emb in embeddings])
    mert_batch = torch.stack([emb[4] for emb in embeddings])
    
    return {
        'emb': (whisper_batch, crepe_batch, chord_batch, beat_this_batch, mert_batch),
    }


def find_song_directories(input_path):
    """Find all song directories containing vocals.wav and accompaniment.wav."""
    input_path = Path(input_path)
    song_dirs = []
    
    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")
    
    # Check if input_path itself is a song directory
    vocals_path = input_path / 'vocals.wav'
    accompaniment_path = input_path / 'accompaniment.wav'
    if vocals_path.exists() and accompaniment_path.exists():
        return [input_path]
    
    # Otherwise, search for song directories within the folder
    if input_path.is_dir():
        for item in input_path.iterdir():
            if item.is_dir():
                vocals_path = item / 'vocals.wav'
                accompaniment_path = item / 'accompaniment.wav'
                if vocals_path.exists() and accompaniment_path.exists():
                    song_dirs.append(item)
    else:
        raise ValueError(f"Input path must be a directory: {input_path}")
    
    return sorted(song_dirs)


def process_single_song(song_dir, data_config, train_config, model_config, model, trainer, batch_size):
    """Process a single song directory and return results."""
    try:
        # Create inference dataset
        dataset = InferenceDataset(song_dir, data_config)
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_fn,
            shuffle=False
        )
        
        # Run inference
        predictions = trainer.predict(model, data_loader)
        
        # Get the prediction result
        if predictions is None or len(predictions) == 0 or predictions[0] is None:
            return None
        
        # Extract the batch result
        result = predictions[0]
        probs = result['probs'].cpu()
        pred = result['predictions'].cpu()
        
        # Get the prediction for the single clip
        if probs.dim() > 1:
            probs = probs[0]  # Take first item if batch size > 1
            pred = pred[0]
        
        return {
            'song_dir': str(song_dir),
            'prediction': pred.item(),
            'probs': probs,
            'confidence': probs.max().item(),
            'real_prob': probs[1].item(),
            'fake_prob': probs[0].item()
        }
    except Exception as e:
        print(f"Error processing {song_dir}: {e}")
        return None


def write_results_csv(results, output_path):
    """Write inference results to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "song_dir",
        "prediction",
        "prediction_label",
        "confidence",
        "real_prob",
        "fake_prob",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "song_dir": result["song_dir"],
                    "prediction": result["prediction"],
                    "prediction_label": "real" if result["prediction"] == 1 else "fake",
                    "confidence": result["confidence"],
                    "real_prob": result["real_prob"],
                    "fake_prob": result["fake_prob"],
                }
            )


def main():
    parser = argparse.ArgumentParser(description='Inference script for AI music model')
    parser.add_argument('input_path', type=str, 
                        help='Path to song directory (containing vocals.wav and accompaniment.wav) or folder containing multiple song directories')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/lennon/AI_music/lightning_logs/spectttra-full-train/checkpoints/epoch=2-step=7980.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='/home/lennon/AI_music/ai_music/configs/SpecTTTra.yaml',
                        help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--csv_out', type=str, default=None,
                        help='Optional path to write inference summary CSV')
    
    args = parser.parse_args()
    
    # Load configs
    with open(args.config) as f:
        configs = yaml.safe_load(f)
        data_config = configs['data']
        train_config = configs['train']
        model_config = configs['model']
    
    # Find all song directories
    print(f"Searching for song directories in: {args.input_path}")
    song_dirs = find_song_directories(args.input_path)
    
    if len(song_dirs) == 0:
        print(f"No valid song directories found in {args.input_path}")
        print("Each song directory must contain vocals.wav and accompaniment.wav")
        return
    
    print(f"Found {len(song_dirs)} song directory(ies)")
    print(f"Processing first {data_config['duration']} seconds of each song...")
    
    # Load model once (shared across all songs)
    print("\nLoading model...")
    print(f"Using config: {args.config}")
    print(f"Using checkpoint: {args.checkpoint}")
    classifier_type = model_config.get('classifier_type', 'ResNet').lower()
    if classifier_type == 'resnet':
        classifier = resnet.ResNet(max_tokens_per_modality=model_config['max_tokens_per_modality'])
    elif classifier_type == 'spectttra':
        classifier = SpecTTTraAttentionClassifier(
            feature_dim=model_config.get('feature_dim'),
            embed_dim=model_config.get('embed_dim'),
            num_heads=model_config.get('num_heads'),
            num_layers=model_config.get('num_layers'),
            tokenizer_clip_size=model_config.get('tokenizer_clip_size'),
            num_classes=2,
            pre_norm=model_config.get('pre_norm'),
            pe_learnable=model_config.get('pe_learnable'),
            pos_drop_rate=model_config.get('pos_drop_rate'),
            attn_drop_rate=model_config.get('attn_drop_rate'),
            proj_drop_rate=model_config.get('proj_drop_rate'),
            mlp_ratio=model_config.get('mlp_ratio'),
        )
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}. Must be 'ResNet' or 'SpecTTTra'")

    model = LightningModel.load_from_checkpoint(
        args.checkpoint,
        classifier=classifier,
        fuser=cross_attention.MultiModalMERTFusion(use_layer_mix=True),
        configs=train_config,
        map_location='cpu'  # Load to CPU first, then move to GPU if available
    )
    model.eval()
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Setup trainer
    trainer = L.Trainer(
        devices=1,
        accelerator="auto",
        precision=train_config.get('precision', '16-mixed'),
    )
    
    # Process each song directory
    results = []
    for i, song_dir in enumerate(song_dirs, 1):
        print(f"\n[{i}/{len(song_dirs)}] Processing: {song_dir}")
        result = process_single_song(song_dir, data_config, train_config, model_config, 
                                     model, trainer, args.batch_size)
        if result:
            results.append(result)
    
    # Print summary results
    print("\n" + "="*60)
    print("INFERENCE RESULTS SUMMARY")
    print("="*60)
    
    if len(results) == 0:
        print("No songs were successfully processed.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['song_dir']}")
        print(f"    Prediction: {'real' if result['prediction'] == 1 else 'fake'}")
        print(f"    Confidence: {result['confidence']:.4f}")
        print(f"    Probabilities - Real: {result['real_prob']:.4f}, Fake: {result['fake_prob']:.4f}")

    if args.csv_out:
        write_results_csv(results, args.csv_out)
        print(f"\nSaved CSV results to: {args.csv_out}")
    
    print("\n" + "="*60)
    print(f"Total processed: {len(results)}/{len(song_dirs)}")
    print("="*60)


if __name__ == "__main__":
    main()
