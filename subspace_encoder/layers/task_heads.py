"""#### *ForMaskedLM"""

# Import the return object, MaskedLMOutput
from typing import Optional
from dataclasses import dataclass

from transformers.modeling_outputs import (
    #BaseModelOutputWithPastAndCrossAttentions,
    #BaseModelOutputWithPoolingAndCrossAttentions,
    #CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    #MultipleChoiceModelOutput,
    #NextSentencePredictorOutput,
    #QuestionAnsweringModelOutput,
    #SequenceClassifierOutput,
    #TokenClassifierOutput,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.shared_space_config import SharedSpaceEncoderConfig
from models.shared_space_encoder import (
    SharedSpaceEncoderPreTrainedModel,
    SharedSpaceEncoderModel
)


# ----------------------------
# BERT-style MLM head pieces
# ----------------------------

class _PredictionHeadTransform(nn.Module):
    """
    Matches BERT's transform block:
        hidden -> Linear(D->D) -> ACT -> LayerNorm(D)
    """
    def __init__(self, config: SharedSpaceEncoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Resolve activation like BERT's ACT2FN, with safe fallbacks
        act = getattr(config, "hidden_act", "gelu")
        if isinstance(act, str):
            act_l = act.lower()
            if act_l in ("gelu", "gelu_new"):   # treat gelu_new same as gelu for simplicity
                self.act_fn = F.gelu
            elif act_l == "relu":
                self.act_fn = F.relu
            elif act_l in ("silu", "swish"):
                self.act_fn = F.silu
            elif act_l == "tanh":
                self.act_fn = torch.tanh
            else:
                raise ValueError(f"Unsupported activation: {act}")
        else:
            # Callable provided in config
            self.act_fn = act
        eps = getattr(config, "layer_norm_eps", 1e-12)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.act_fn(x)
        x = self.LayerNorm(x)
        return x


class _SharedSpaceLMPredictionHead(nn.Module):
    """
    BERT-like LM head that:
      1) applies the transform block,
      2) (optionally) projects to vocab latent subspace via encoder_model.vocab_proj,
      3) decodes with a tied Linear to vocab embeddings and adds a learned bias.
    """
    def __init__(self, config: SharedSpaceEncoderConfig, encoder_model: SharedSpaceEncoderModel):
        super().__init__()
        self.transform = _PredictionHeadTransform(config)
        self.encoder_model = encoder_model  # to access vocab_embed and optional vocab_proj

        # Determine embedding (decoder input) dimension from the tied embeddings.
        # If you use a subspace, vocab_embed.embedding_dim should be C; otherwise D.
        emb_dim = encoder_model.vocab_embed.embedding_dim  # C or D
        self.decoder = nn.Linear(emb_dim, config.vocab_size, bias=False)

        # Per-token bias like BERT
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias  # keep the link so bias resizes together

        # Tie decoder weights to input embeddings *when shapes match*
        # (For subspace tying, vocab_embed.weight is [V, C]; otherwise [V, D].)
        self._tie_decoder_to_embeddings()

    def _tie_decoder_to_embeddings(self) -> None:
        dec_w = self.decoder.weight      # [V, C or D]
        emb_w = self.encoder_model.vocab_embed.weight  # [V, C or D]
        if dec_w.shape == emb_w.shape:
            # Make them the same parameter reference (true tying).
            self.decoder.weight = emb_w
        else:
            # Fallback: shapes differ (shouldn't happen if emb_dim was taken from vocab_embed)
            # We leave them untied to avoid shape errors.
            pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            hidden_states: [B, T, D]
        Returns:
            logits: [B, T, V]
        """
        x = self.transform(hidden_states)  # [B, T, D]

        # Optional projection into vocab latent space (C) before decoding.
        # Expect encoder_model.vocab_proj: Linear(D -> C) if present.
        vocab_proj = getattr(self.encoder_model, "vocab_proj", None)
        if vocab_proj is not None:
            x = vocab_proj(x)  # [B, T, C]

        logits = self.decoder(x)  # [B, T, V] (decoder tied to vocab_embed)
        return logits


class SharedSpaceEncoderForMaskedLM(SharedSpaceEncoderPreTrainedModel):
    """
    Refactored to mirror BERT's MLM head:
      - Transform: Linear(D->D) + activation + LayerNorm
      - Optional vocab subspace projection via `encoder_model.vocab_proj` (D->C)
      - Decoder: Linear(C or D -> V), weight-tied to `encoder_model.vocab_embed.weight`
      - Plus a learned output bias (per token), as in BERT
    """

    def __init__(self, config: SharedSpaceEncoderConfig) -> None:
        super().__init__(config)
        self.encoder_model = SharedSpaceEncoderModel(config)
        self.cls = _SharedSpaceLMPredictionHead(config, self.encoder_model)
        self.post_init()

    # These two help HF-style resizing and weight tying workflows, if you use them.
    def get_output_embeddings(self):
        return self.cls.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.cls.decoder = new_embeddings
        # reattach bias link to keep resize semantics
        self.cls.decoder.bias = self.cls.bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        """
        Inputs:
               input_ids:      [B, T]
          attention_mask:      broadcastable mask as expected by encoder_model
                 labels:       [B, T], with -100 for non-masked positions

        Outputs:
               logits:         [B, T, V]
                 loss:         scalar (mean over masked positions) if labels provided, else None
        """
        # Run encoder; expected to return last hidden state [B, T, D]
        hidden_states = self.encoder_model(
            input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # If your SharedSpaceEncoderModel returns a dataclass, uncomment:
        # hidden_states = hidden_states.last_hidden_state

        # BERT-style head (transform + optional subspace + tied decoder + bias)
        logits = self.cls(hidden_states)  # [B, T, V]

        loss = None
        if labels is not None:
            # Cross-entropy over masked positions (-100 ignored).
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


"""#### `*ForSequenceClassification`

Copied from BERT
"""

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput

# Import Union
from typing import Union

class SharedSpaceEncoderForSequenceClassification(SharedSpaceEncoderPreTrainedModel):
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Call the `*PreTrainedModel` init.
        super().__init__(config)

        # Create the `*Model`. Everything we need is already there.
        self.encoder_model = SharedSpaceEncoderModel(config)

        # Incorporate BERT's pooling layer.
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)


        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        #token_type_ids: Optional[torch.Tensor] = None,
        #position_ids: Optional[torch.Tensor] = None,
        #head_mask: Optional[torch.Tensor] = None,
        #inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        # ============================
        #        Evaluate
        # ===========================
        hidden_states = self.encoder_model(
            input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            #position_ids=position_ids,
            #head_mask=head_mask,
            #inputs_embeds=inputs_embeds,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
        )

        # ===============================
        #      Non-Linearity ("Pooler")
        # ==============================
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # ====================
        #      Classifier
        # ====================
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            # TODO
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )



