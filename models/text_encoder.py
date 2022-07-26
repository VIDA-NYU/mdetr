from typing import List, Optional, cast
from torch import Tensor, nn
from transformers import RobertaModel, RobertaTokenizerFast


class HFTextEncoder(nn.Module):
    
    def __init__(self, d_model, text_encoder_type=None, freeze_text_encoder=False, expander_dropout=0.1):
        super().__init__()
        text_encoder_type = text_encoder_type or self.text_encoder_type
        Model, Tokenizer = self._get_model_types()
        self.tokenizer = Tokenizer.from_pretrained(text_encoder_type)
        self.encoder = cast(Model, Model.from_pretrained(text_encoder_type))
        self.config = self.encoder.config

        if freeze_text_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        print(self.encoder.config.hidden_size, d_model)
        self.resizer = FeatureResizer(
            input_feat_size=self.encoder.config.hidden_size,
            output_feat_size=d_model,
            dropout=expander_dropout,
        )

    def _get_model_types(self):
        raise NotImplementedError

    def forward(self, text, device):
        if not isinstance(text[0], str):
            # The text is already encoded, use as is.
            text_attention_mask, text_memory_resized, tokenized, pooled_encoded_text = text
            return text_attention_mask, text_memory_resized, tokenized, pooled_encoded_text

        # Encode the text
        tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(device)
        encoded_text = self.encoder(**tokenized)
        pooled_text = encoded_text.pooler_output
        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = self.resizer(text_memory)
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        print(text_attention_mask.shape)
        print(text_memory_resized.shape)
        print(type(tokenized))
        print(pooled_text.shape)
        # text_attention_mask: [batch, seq]
        # text_memory_resized: [seq, batch, resized_embedding]
        # tokenized: tokenization_utils_base.BatchEncoding
        # pooled_text: [batch, encoder_embedding]
        return text_attention_mask, text_memory_resized, tokenized, pooled_text



class RobertaTextEncoder(HFTextEncoder):
    text_encoder_type = "roberta-base"
    def _get_model_types(self):
        from transformers import RobertaModel, RobertaTokenizerFast
        return RobertaModel, RobertaTokenizerFast



# https://huggingface.co/docs/transformers/model_doc/clip#usage
# https://github.com/huggingface/transformers/blob/d3cb28886ac68beba9a6646b422a4d727b056c0c/src/transformers/models/clip/modeling_clip.py
class ClipTextEncoder(HFTextEncoder):
    text_encoder_type = "openai/clip-vit-base-patch32"
    def _get_model_types(self):
        from transformers import CLIPTextModel, CLIPTokenizerFast
        return CLIPTextModel, CLIPTokenizerFast


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output
