from typing import List, Optional, cast
from torch import Tensor, nn
from transformers import RobertaModel, RobertaTokenizerFast




class RobertaTextEncoder(nn.Module):
    def __init__(self, d_model, text_encoder_type="roberta-base", freeze_text_encoder=False, expander_dropout=0.1):
        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        self.text_encoder = cast(RobertaModel, RobertaModel.from_pretrained(text_encoder_type))

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.resizer = FeatureResizer(
            input_feat_size=self.text_encoder.config.hidden_size,
            output_feat_size=d_model,
            dropout=expander_dropout,
        )

    def forward(self, text, device):
        if not isinstance(text[0], str):
            # The text is already encoded, use as is.
            text_attention_mask, text_memory_resized, tokenized = text
            return text_attention_mask, text_memory_resized, tokenized, None

        # Encode the text
        tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(device)
        encoded_text = self.text_encoder(**tokenized)
        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = self.resizer(text_memory)
        return text_attention_mask, text_memory_resized, tokenized, encoded_text



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
