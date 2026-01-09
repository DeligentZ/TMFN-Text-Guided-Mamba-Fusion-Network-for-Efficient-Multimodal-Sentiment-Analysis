import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel,Data2VecAudioModel, BertModel, ViTModel, AutoTokenizer
from models.cross_attn_encoder import BertConfig, MambaLayer
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dataset = config["dataset_name"]
        # 1. pretrain model config
        self.text_model = self._pretrained_model_text(dataset)
        self.audio_model = self._pretrained_model_audio(dataset)
        self.video_model = self._pretrained_model_video(self.config["video_pretrained_model"])
        # 2. fusion model config
        self.Mamba_layers = MambaLayer(BertConfig())
        # 5. output layer
        self.output_layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(512, 1)
        )

    def forward(self,text_inputs, text_mask, audio_inputs, audio_mask, video_inputs):
        raw_text = self.text_model(text_inputs, text_mask)
        text_hidden = raw_text.last_hidden_state # [Batch_size, Seq_length, Hidden_size]
        raw_audio = self.audio_model(audio_inputs, audio_mask, output_attentions=True)
        audio_hidden = raw_audio.last_hidden_state # [Batch_size, Seq_length, Hidden_size]
        new_audio_mask = self._audio_mask(audio_hidden, audio_mask)
        video_feature, video_mask = self._video_feature_extract(video_inputs)
        fused_feature = self.Mamba_layers(text_hidden, text_mask, audio_hidden, new_audio_mask, video_feature, video_mask)
        output_layer = self.output_layers(fused_feature)
        return output_layer

    def _pretrained_model_text(self, dataset):
        if dataset == "sims":
            # load ch text pretrained model
            if self.config["ch_text_pretrained_model"] == "macbert-base":
                print("ch_text_pretrained_model: macbert-base")
                model = BertModel.from_pretrained("hfl/chinese-macbert-base")
                self._freeze_models(model)
                self._unfreeze_modules(model.encoder.layer, 2)
                return model
            else:
                model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
                print("ch_text_pretrained_model: hfl/chinese-roberta-wwm-ext")
                self._freeze_models(model)
                self._unfreeze_modules(model.encoder.layer, 2)
                return model
        else:
            # load en text pretrained model
            if self.config["en_text_pretrained_model"] == "deberta-v3":
                print("en_text_pretrained_model:deberta-v3")
                model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
                self._freeze_models(model)
                self._unfreeze_modules( model.deberta.encoder.layer, 2)
                return model
            else:
                model = RobertaModel.from_pretrained('roberta-base')
                print("en_text_pretrained_model:roberta-base")
                self._freeze_models(model)
                self._unfreeze_modules(model.encoder.layer, 2)
                return model

    def _pretrained_model_audio(self, dataset):
        if dataset == "sims":
            # load ch text pretrained model
            if self.config["ch_audio_pretrained_model"] == "Data2Vec-audio-base":
                print("ch_audio pretrained_model: Data2Vec-audio-base")
                model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
                self._freeze_models(model)
                self._unfreeze_modules(model.encoder.layers, 2)
                return model
            else:
                print("ch_audio pretrained_model: TencentGameMate/chinese-hubert-base")
                model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-base")
                self._freeze_models(model)
                self._unfreeze_modules(model.encoder.layers, 2)
                return model
        else:
            # load en audio pretrained model
            if self.config["en_audio_pretrained_model"] == "facebook/data2vec-audio-base":
                print("en_audio_pretrained_model:data2vec-audio-base")
                model =  Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
                self._freeze_models(model)
                self._unfreeze_modules(model.encoder.layers, 2)
                return model
            else:
                print("en_audio_pretrained_model:facebook/hubert-base-ls960")
                model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
                self._freeze_models(model)
                self._unfreeze_modules(model.encoder.layers, 2)
                return model

    def _pretrained_model_video(self, pretrained_model_name):
        # load video pretrained model
        if pretrained_model_name == "vit-base":
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

            self._freeze_models(model)
            self._unfreeze_modules(model.encoder.layer, 2)
            return model
        else:
            model = ViTModel.from_pretrained("facebook/deit-base-patch16-224")
            self._freeze_models(model)
            self._unfreeze_modules(model.encoder.layer, 2)
            return model

    def _video_feature_extract(self, video_inputs):
        B, T, C, H, W = video_inputs.shape
        video_tensor = video_inputs.view(B * T, C, H, W)
        outputs = self.video_model(video_tensor)
        cls_token = outputs.pooler_output
        video_features = cls_token.view(B, T, -1)
        video_mask = torch.ones(B, T, dtype=torch.float).to(device)
        return video_features, video_mask

    def _freeze_models(self, model):
        for param in model.parameters():
            param.requires_grad = False # False

    def _unfreeze_modules(self, layers, num_layer = 2):
        for p in layers[-num_layer:].parameters():
            p.requires_grad = True

    def _audio_mask(self,audio_hidden, audio_mask):
        B, L_hidden, D = audio_hidden.shape  # output of encoder
        audio_mask_new = F.interpolate(
            audio_mask.unsqueeze(1),  # [B,1,L_input]
            size=L_hidden,
            mode='nearest'
        ).squeeze(1).float().to(device)  # [B, L_hidden]
        return audio_mask_new





