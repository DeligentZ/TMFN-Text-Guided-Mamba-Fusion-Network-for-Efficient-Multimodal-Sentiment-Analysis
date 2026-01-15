import torch
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Dataset_sims(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_directory,ch_text_tokenizer_model, mode):
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].reset_index()
        # store labels
        self.targets = df['label']
        # store texts
        self.texts = df['text']
        if ch_text_tokenizer_model == "chinese-hubert-base":
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
        # store audio
        self.audio_file_paths = []
        # store video
        self.video_file_paths = []
        self.sample_ids = []
        for i in range(0,len(df)):
            clip_id = str(df['clip_id'][i]).zfill(4)
            video_id = str(df['video_id'][i])

            audio_path = os.path.join(audio_directory, video_id, f"{clip_id}.wav")
            video_path = os.path.join(audio_directory, video_id, f"{clip_id}.pt")

            self.audio_file_paths.append(audio_path)
            self.video_file_paths.append(video_path)
            self.sample_ids.append(f"{video_id}_{clip_id}")
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    def __getitem__(self, index):
        #extract text features
        text = str(self.texts[index])
        sound, samplerate = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)
        tokenized_text = self.tokenizer(
            text,            
            max_length = 96,
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        )               
                
        # extract audio features    

        features = self.feature_extractor(
            soundData,
            sampling_rate=16000,
            max_length=64000,
            return_attention_mask=True,
            truncation=True,
            padding="max_length"
        )
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.float).squeeze()
        # extract video features
        video_features = torch.load(self.video_file_paths[index])
        return { # text
                "text_tokens": tokenized_text["input_ids"],
                "text_masks": tokenized_text["attention_mask"],
                 # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                 # video
                "video_features": video_features,
                 # labels
                "targets":  self.targets[index],
                "ids": self.sample_ids[index]
                }
    
    def __len__(self):
        return len(self.targets)

def collate_fn_sims(batch):
    text_tokens = []
    text_masks = []
    audio_inputs = []
    audio_masks = []

    targets = []
    video_inputs = []

    # organize batch
    for i in range(len(batch)):
        # text
        text_tokens.append(batch[i]['text_tokens'])
        text_masks.append(batch[i]['text_masks'])
        # audio
        audio_inputs.append(batch[i]['audio_inputs'])
        audio_masks.append(batch[i]['audio_masks'])
        # video
        video_inputs.append(batch[i]['video_features'])
        # labels
        targets.append(batch[i]['targets'])

    return {
        # text
        "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
        "text_masks": torch.tensor(text_masks, dtype=torch.float),
        # audio
        "audio_inputs": torch.stack(audio_inputs),
        "audio_masks": torch.stack(audio_masks),
        # video
        "video_features": torch.stack(video_inputs),
        # labels
        "targets": torch.tensor(targets, dtype=torch.float32),
    }

class Dataset_mosei(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_directory, en_text_tokenizer_model, mode, ur = False):
        df = pd.read_csv(csv_path)
        self.ur = ur
        if ur is False:
            df = df[df['mode'] == mode].sort_values(by=['video_id', 'clip_id']).reset_index()
        else:
            df = df[df['mode'] == mode].reset_index()
        # store labels
        self.targets_M = df['label']
        # store texts
        self.texts = df['text'].apply(lambda x: x[0].upper() + x[1:].lower() if len(x) > 0 else x).values
        #tokenizer
        if en_text_tokenizer_model == "roberta-base":
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        # store audio and video
        self.audio_file_paths = []
        self.video_file_paths = []
        self.sample_name = []
        ## loop through the csv entries
        if ur is False:
            for i in range(0,len(df)):
                file_name_audio = str(df['video_id'][i])+'/'+str(df['clip_id'][i])+'.wav'
                file_path_audio = audio_directory + "/" + file_name_audio
                file_name_video = str(df['video_id'][i]) + '/' + str(df['clip_id'][i]) + '.pt'
                file_path_video = audio_directory + "/" + file_name_video
                sample_name = str(df['video_id'][i]) + '_' + str(df['clip_id'][i])
                self.audio_file_paths.append(file_path_audio)
                self.video_file_paths.append(file_path_video)
                self.sample_name.append(sample_name)
            self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        else:
            for i in range(0, len(df)):
                file_name_audio = str(df['sample_id'][i])+ '.wav'
                file_path_audio = os.path.join(audio_directory, file_name_audio)
                file_name_video = str(df['sample_id'][i]) + '.npy'
                file_path_video = os.path.join(audio_directory, file_name_video)
                self.audio_file_paths.append(file_path_audio)
                self.video_file_paths.append(file_path_video)
                self.sample_name.append(df['sample_id'][i])
            self.feature_extractor = self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        
    def __getitem__(self, index):
        text = str(self.texts[index])
        if self.ur is False:
            sound, sample_rate = torchaudio.load(self.audio_file_paths[index])
            soundData = torch.mean(sound, dim=0, keepdim=False)
            # tokenize text
            tokenized_text = self.tokenizer(
                    text,
                    max_length = 96,
                    padding = "max_length",     # Pad to the specified max_length.
                    truncation = True,          # Truncate to the specified max_length.
                    add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.
                    return_attention_mask = True
                )
             # extract audio features
            features = self.feature_extractor(
                soundData,
                sampling_rate=16000,
                max_length=64000,
                return_attention_mask=True,
                truncation=True,
                padding="max_length"
            )
            audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
            audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.float).squeeze()

            #extract video features
            video_features = torch.load(self.video_file_paths[index])
            sample_id = self.sample_name[index]
        else:
            sound, sample_rate = torchaudio.load(self.audio_file_paths[index])
            soundData = torch.mean(sound, dim=0, keepdim = False)
            tokenized_text = self.tokenizer(
                text,
                max_length=192,
                padding="max_length",  # Pad to the specified max_length.
                truncation=True,  # Truncate to the specified max_length.
                add_special_tokens=True,  # Whether to insert [CLS], [SEP], <s>, etc.
                return_attention_mask=True
            )
            features = self.feature_extractor(
                soundData,
                sampling_rate=16000,
                max_length=240000,
                return_attention_mask=True,
                truncation=True,
                padding="max_length"
            )
            audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
            audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.float).squeeze()

            # extract video features
            if self.ur is False:
                video_features = torch.load(self.video_file_paths[index])
            else:

                video_features = torch.from_numpy(np.load(self.video_file_paths[index]))
            sample_id = self.sample_name[index]

        return { # text
                "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
                "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.float),
                # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks.float(),
                # video
                "video_features": video_features,
                 # labels
                "targets": torch.tensor(self.targets_M[index], dtype=torch.float),
                "ids": sample_id
                }

    def __len__(self):
        return len(self.targets_M)

def data_loader(config, generator=None):
    dataset = config["dataset_name"]
    batch_size = config["batch_size"]
    ch_text_tokenizer_model = config["ch_text_pretrained_model"]
    en_text_tokenizer_model = config["en_text_pretrained_model"]
    if dataset == 'mosei':
        csv_path = '/MOSEI/label.csv'
        audio_file_path = "/MOSEI/wav"
        train_data = Dataset_mosei(csv_path, audio_file_path, en_text_tokenizer_model,'train', ur=False)
        test_data = Dataset_mosei(csv_path, audio_file_path, en_text_tokenizer_model,'test', ur=False)
        val_data = Dataset_mosei(csv_path, audio_file_path, en_text_tokenizer_model,'valid', ur=False)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
        return train_loader, val_loader, test_loader
    elif dataset == 'ur_funny':
        csv_path = 'label.csv'
        audio_file_path = "/urfunny_processed_251202/wav"
        train_data = Dataset_mosei(csv_path, audio_file_path, en_text_tokenizer_model, 'train', ur=True)
        test_data = Dataset_mosei(csv_path, audio_file_path, en_text_tokenizer_model, 'test', ur=True)
        val_data = Dataset_mosei(csv_path, audio_file_path, en_text_tokenizer_model, 'valid', ur=True)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
        return train_loader, val_loader, test_loader

    else:
        csv_path = '/SIMS/label.csv'
        audio_file_path = "/SIMS/wav"
        train_data = Dataset_sims(csv_path, audio_file_path, ch_text_tokenizer_model,'train')
        test_data = Dataset_sims(csv_path, audio_file_path, ch_text_tokenizer_model,'test')
        val_data = Dataset_sims(csv_path, audio_file_path, ch_text_tokenizer_model,'valid')
        
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=True, pin_memory=True,num_workers=6)
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False,  pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False, pin_memory=True)
        return train_loader, val_loader, test_loader
