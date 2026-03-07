import sys
sys.path.insert(0, str("/home/lennon/AI_music/ISMIR2019-Large-Vocabulary-Chord-Recognition"))
# sys.path.insert(0, str("/home/lennon/AI_music/beat_this"))
sys.path.insert(0, str("/home/lennon/AI_music/beat_this/beat_this")) # different path for submodule setup
sys.path.insert(0, "/home/lennon/AI_music")
from beat_this.inference import load_model, LogMelSpect
from feature_extractor import FeatureExtractor

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch

from torch.utils.data import DataLoader
import numpy as np
import torchaudio
import torchcrepe
import whisper
import pandas as pd
from pathlib import Path
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

class AudioDataset():
    def __init__(self, data_configs, split):
        self.paths_csv = Path(data_configs["data_root"])
        self.sr = data_configs["sample_rate"]
        self.duration = data_configs["duration"]
        self.random_sample = data_configs["random_sample"]
        self.whisper_size = data_configs["whisper_size"]
        self.crepe_size = data_configs["crepe_size"]
        
        self.whisper = None
        self.chordnet = None
        self.beat_this = None
        self.bt_spec_extractor = None
        self.mert_model = None
        self.mert_processor = None
        self.tracks = None
        self._models_initialized = False

        self.pathext = Path(os.path.split(self.paths_csv)[0])
        df = pd.read_csv(self.paths_csv)
        mask = df.apply(lambda row: file_pair_exists(row, self.pathext), axis=1)
        self.df = df[mask].reset_index(drop=True)
        self.get_tracks(split)
        
        self._init_models()
        print(split, "size:", self.__len__())
    
    def _init_models(self):
        if self._models_initialized:
            return
        if self.whisper is None:
            self.whisper = whisper.load_model(self.whisper_size, device='cuda')
        if self.chordnet is None:
            self.chordnet = FeatureExtractor()
        if self.beat_this is None:
            self.beat_this = load_model('/home/lennon/AI_music/beat_this/final0.ckpt', device='cuda')
            self.bt_spec_extractor = LogMelSpect(
                sample_rate=22050,
                n_fft=1024,
                hop_length=441,
                n_mels=128,
                device='cuda'
            )
        if self.mert_model is None:
            self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
            self.mert_model = self.mert_model.to('cuda')
            self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        
        self._models_initialized = True

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        if not self._models_initialized:
            self._init_models()
            
        idx_row = self.tracks.iloc[idx]
        v_path = self.pathext / idx_row['source'] / idx_row['filename'] / 'vocals.wav'
        a_path = self.pathext / idx_row['source'] / idx_row['filename'] / 'accompaniment.wav'

        v_audio, sr = torchaudio.load(v_path)
        a_audio, sr = torchaudio.load(a_path)
        if v_audio.shape[0] > 1:
            v_audio = v_audio.float().mean(dim=0, keepdim=True)
            a_audio = a_audio.float().mean(dim=0, keepdim=True)

        if self.sr and (sr != self.sr):
            transform = torchaudio.transforms.Resample(sr, self.sr)
            v_audio = transform(v_audio)
            a_audio = transform(a_audio)
            sr = self.sr
        duration = self.duration * sr

        if len(v_audio[0]) < duration or len(v_audio[0]) != len(a_audio[0]):
            return None

        audio_start = self.random_sample * np.random.randint(0, len(v_audio[0])-duration+1)
        v_clip = v_audio[:, audio_start:audio_start+duration]
        a_clip = a_audio[:, audio_start:audio_start+duration]

        whisper_emb = self._lyrics_emb(v_clip)
        crepe_emb = self._pitch_emb(v_clip)
        chord_emb = self._chord_emb(a_clip)
        beat_emb = self._beat_emb(a_clip)
        mert_emb = self._mert_emb(a_clip)

        # Move embeddings to CPU for dataloader (pin_memory requires CPU tensors)
        embeddings = (whisper_emb.detach().cpu(), crepe_emb.detach().cpu(), chord_emb.detach().cpu(), 
                      beat_emb.detach().cpu(), mert_emb.detach().cpu())
        sample = {"emb": embeddings,
                  "label": idx_row['source']}

        return sample
    
    def _pitch_emb(self, clip):
        # "tiny"=32, "full"=64
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
        for chunk in audio_chunks(clip, self.sr*30):
            mel = whisper.log_mel_spectrogram(chunk.squeeze(0))  # (80, 500)
            with torch.no_grad():
                result = self.whisper.encoder(mel.unsqueeze(0).to('cuda'))
                whisper_embs.append(result)
        return torch.cat(whisper_embs, dim=1)

    def _chord_emb(self, clip):
        chordnet = self.chordnet.extract_features_from_audio(clip.squeeze(0), self.sr)
        return torch.from_numpy(chordnet.astype(np.float32, copy=False)).unsqueeze(0)
    
    def _beat_emb(self, clip):
        # beat_this model expects 22050 Hz
        if self.sr != 22050:
            resampler = torchaudio.transforms.Resample(self.sr, 22050)
            clip = resampler(clip)
        
        spect = self.bt_spec_extractor(clip.to('cuda'))
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
    
    def get_tracks(self, split):
        split_ratio = 0.9

        real_df = self.df[self.df['source'] == 'real'].sample(frac=1, random_state=42).reset_index(drop=True)
        fake_df = self.df[self.df['source'] == 'fake'].sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print(f"Total dataset - Real: {len(real_df)}, Fake: {len(fake_df)}")
        
        real_split_index = int(split_ratio * len(real_df))
        real_train = real_df[:real_split_index]
        real_val = real_df[real_split_index:]
        
        fake_split_index = int(split_ratio * len(fake_df))
        fake_train = fake_df[:fake_split_index]
        fake_val = fake_df[fake_split_index:]
        
        if split == "train":
            self.tracks = pd.concat([real_train, fake_train], ignore_index=True)
            self.tracks = self.tracks.sample(frac=1, random_state=42).reset_index(drop=True)
            
        elif split == "val":
            self.tracks = pd.concat([real_val, fake_val], ignore_index=True)
            self.tracks = self.tracks.sample(frac=1, random_state=42).reset_index(drop=True)


def file_pair_exists(row, basepath):
    song_dir = basepath / row["source"] / str(row["filename"])
    vocals = song_dir / "vocals.wav"
    accomp = song_dir / "accompaniment.wav"
    return vocals.exists() and accomp.exists()

def audio_chunks(clip, chunk_length):
    n_samples = clip.shape[1]
    n_chunks = n_samples // chunk_length
    trimmed = clip[:, :n_chunks * chunk_length]

    chunks = torch.split(trimmed, chunk_length, dim=1)
    return chunks

def collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    embeddings = [item['emb'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Stack each embedding type across the batch
    whisper_batch = torch.stack([emb[0] for emb in embeddings])
    crepe_batch = torch.stack([emb[1] for emb in embeddings])
    chord_batch = torch.stack([emb[2] for emb in embeddings])
    beat_this_batch = torch.stack([emb[3] for emb in embeddings])
    mert_batch = torch.stack([emb[4] for emb in embeddings])
    
    return {
        'emb': (whisper_batch, crepe_batch, chord_batch, beat_this_batch, mert_batch),
        'label': labels
    }

def get_dataloader(split, data_configs, train_configs, shuffle):

    dataset = AudioDataset(data_configs, split)

    dataloader = DataLoader(
        dataset,
        batch_size=train_configs['batch_size'],
        num_workers=train_configs['num_workers'],
        pin_memory=train_configs['pin_memory'],
        collate_fn=collate,
        shuffle=shuffle
    )

    return dataloader
