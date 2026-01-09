import torch
from torch import nn
import math
from mamba_ssm import Mamba
import torch.nn.functional as F
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=1,
                 num_attention_heads=12,
                 intermediate_size=2304,
                 hidden_act="relu",
                 hidden_dropout_prob=0.2,
                 attention_probs_dropout_prob = 0.2,
                 max_position_embeddings=512,
                 add_abs_pos_emb = False,
                 add_pos_enc = False):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.add_abs_pos_emb = add_abs_pos_emb
        self.add_pos_enc = add_pos_enc

BertLayerNorm = torch.nn.LayerNorm

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size


        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.add_abs_pos_emb = config.add_abs_pos_emb
        if self.add_abs_pos_emb:
            self.abs_pos_emb = nn.Parameter(torch.randn(512, self.attention_head_size))
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)#(b, s,h, d) -> (b, h, s, d)

    def forward(self, hidden_states, context, attention_mask=None):
        #print(context.size(),attention_mask.size())
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        if self.add_abs_pos_emb:
            pos_emb = self.abs_pos_emb[0:context.size(1),:]
            pos_emb_q = self.abs_pos_emb[0:hidden_states.size(1),:]
            pos_emb_q = pos_emb_q.expand(query_layer.size(0), query_layer.size(1), -1, -1)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # shape is (b, h, s_q, s_k)
        if self.add_abs_pos_emb:
            attention_pos_scores = torch.matmul(query_layer+pos_emb_q, pos_emb.transpose(-1, -2))
            attention_scores = (attention_scores+attention_pos_scores) / math.sqrt(self.attention_head_size)
        else:
            attention_scores = attention_scores/ math.sqrt(self.attention_head_size)

        # Apply the attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand((-1,attention_scores.size(1),attention_scores.size(2),-1))
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
            #print(attention_mask.size())
            #print(attention_scores.size())
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # shape is (b, h, s_q, d)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # shape is (b, s_q, h, d)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor) # attention_output = self.output(output, input_tensor)

        return attention_output

class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).

        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""
class ModalityAdapter(nn.Module):
    def __init__(self, hidden_dim=768, bottleneck=96, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, bottleneck)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return residual +x
class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=64, d_conv=4, expand=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))
class MambaConcatFusion(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),#3
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        self.final_mamba = MambaBlock(d_model=hidden_size, dropout=dropout)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
    def forward(self, text, audio, video):
         fused = torch.cat([text, audio, video], dim=2)
         fused = self.projection(fused)  # [B, T_t, 768]
         fused = self.final_mamba(fused)  # [B, T_t, 768]
         fused = fused.mean(dim = 1)
         return fused
class GateResidual(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.gate_audio = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.gate_image = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.ln_after_gate_audio = nn.LayerNorm(768)
        self.ln_after_gate_image = nn.LayerNorm(768)
    def forward(self,text, audio, image):
        gate_aud = self.gate_audio(text)
        gate_img = self.gate_image(text)
        text_aud_gat_res = gate_aud * audio + (1 - gate_aud) * text
        text_aud_gat_res = self.ln_after_gate_audio(text_aud_gat_res)
        text_img_gat_res = gate_img * image + (1 - gate_img) * text
        text_img_gat_res = self.ln_after_gate_image(text_img_gat_res)
        return text_aud_gat_res, text_img_gat_res
class MambaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Adapter init
        self.text_adapter = ModalityAdapter(config.hidden_size,  96, config.hidden_dropout_prob)
        self.audio_adapter = ModalityAdapter(config.hidden_size, 96, config.hidden_dropout_prob)
        self.image_adapter = ModalityAdapter(config.hidden_size, 96, config.hidden_dropout_prob)
        # Mamba init
        self.text_mamba = MambaBlock(d_model=768, dropout=config.hidden_dropout_prob)
        self.audio_mamba = MambaBlock(d_model=768, dropout=config.hidden_dropout_prob)
        self.image_mamba = MambaBlock(d_model=768, dropout=config.hidden_dropout_prob)
        # CrossAttn
        self.cross_attn_audio = BertCrossattLayer(config)
        self.cross_attn_image = BertCrossattLayer(config)
        self.gate= GateResidual(hidden_size=768, dropout=config.hidden_dropout_prob)
        self.fusion = MambaConcatFusion(hidden_size=768, dropout=config.hidden_dropout_prob)
    def forward(self, text, text_mask, audio, audio_mask, image, image_mask):
        #add adapter
        text = self.text_adapter(text)       #[batch_size, 96 , 768]
        audio = self.audio_adapter(audio)    #[batch_size, 150, 768]
        image = self.image_adapter(image)    #[batch_size, 8, 768]
        # mamba
        text_mamba = self.text_mamba(text) #[batch_size, 96 , 768]
        audio_mamba = self.audio_mamba(audio) # [batch_size, 150, 768]
        image_mamba = self.image_mamba(image) # [batch_size, 8, 768]
        # cross attn
        text_aud = self.cross_attn_audio(input_tensor = text_mamba, ctx_tensor = audio_mamba, ctx_att_mask = audio_mask )
        text_img = self.cross_attn_image(input_tensor = text_mamba, ctx_tensor = image_mamba, ctx_att_mask = image_mask)
        text_aud_gat_res, text_img_gat_res = self.gate(text_mamba, text_aud, text_img)
        fused = self.fusion(text_mamba, text_aud_gat_res, text_img_gat_res)
        return fused

































