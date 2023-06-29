import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from typing import Optional, Tuple
import torch.nn.functional as F

from ernie_m import ErnieMModel, ErnieMPreTrainedModel
from sparse_attn import TransformerSeqLayer
from utils import  seq_id

@dataclass
class UIEModelOutput(ModelOutput):
    """
    Output class for outputs of UIE.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_prob (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (after Sigmoid).
        end_prob (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (after Sigmoid).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss_bce: Optional[torch.FloatTensor] = None
    loss_fsl: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    fuzzy_span_attentions: Optional[Tuple[torch.FloatTensor]] = None

class UIEM(ErnieMPreTrainedModel):
    """
    UIE model based on Bert model.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`PretrainedConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: PretrainedConfig):
        super(UIEM, self).__init__(config)
        self.encoder = ErnieMModel(config)
        self.config = config
        hidden_size = self.config.hidden_size

        adapt_span_params = {'adapt_span_enabled': True, 'adapt_span_loss': 0.0, 'adapt_span_ramp': 32,
                             'adapt_span_init': 0.0, 'adapt_span_cache': False}
        self.sparse_attn_layer = TransformerSeqLayer(
                                                    hidden_size=hidden_size, 
                                                    nb_heads=8, attn_span=30,dropout=0.1,
                                                    inner_hidden_size=hidden_size, adapt_span_params=adapt_span_params)

        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.fc_start = nn.Linear(hidden_size, hidden_size)
        self.fc_end = nn.Linear(hidden_size, hidden_size)
        self.BCE_loss=nn.BCEWithLogitsLoss()
        self.KL_loss=nn.KLDivLoss()
        self.dropout_layer=nn.Dropout()
        self.post_init()


    def forward(self, input_ids: Optional[torch.Tensor] = None,
                input_embedding: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = True,
                
                ):
        """
        Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            input_embedding=input_embedding,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        sequence_output = outputs.last_hidden_state
        attentions=outputs.attentions
        if output_attentions:
            sequence_output,fuzzy_span_attentions = self.sparse_attn_layer(sequence_output,output_attentions=output_attentions)
        else:
            sequence_output = self.sparse_attn_layer(sequence_output,output_attentions=output_attentions)
            fuzzy_span_attentions=None
        start_logits = self.linear_start(
                            self.dropout_layer(
                                self.relu(
                                        self.fc_start(sequence_output)
                                        )
                                            )
        )
        start_prob = torch.squeeze(start_logits, -1)

        end_logits = self.linear_end(
                            self.dropout_layer(
                                self.relu(
                                        self.fc_end(sequence_output)
                                        )
                                            )
        )

        end_prob = torch.squeeze(end_logits, -1)

        start_prob[attention_mask==0]=-10000
        end_prob[attention_mask==0]=-10000


        total_loss = 0.0

        start_prob_copy = start_prob.clone()
        end_prob_copy = end_prob.clone()
        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.float()
            end_positions = end_positions.float()

            #computing the base loss:
            start_loss = self.BCE_loss(start_prob_copy, start_positions)
            end_loss = self.BCE_loss(end_prob_copy, end_positions)
            loss_base = (start_loss + end_loss) / 2.0

            #computing the fsl:
            start_fs=seq_id(start_positions).to(start_positions.device)
            end_fs=seq_id(end_positions).to(end_positions.device)

            loss_fsl_start = self.KL_loss(F.log_softmax(start_prob_copy), start_fs)
            loss_fsl_end = self.KL_loss(F.log_softmax(end_prob_copy), end_fs)
            loss_fsl = (loss_fsl_start + loss_fsl_end) / 2.0
        else:
            loss_base=None
            loss_fsl=None

        #computing the pred
        start_prob=self.sigmoid(start_prob_copy)
        end_prob=self.sigmoid(end_prob_copy)

        if not return_dict:
            output = (start_prob, end_prob) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return UIEModelOutput(
            loss_bce=loss_base,
            loss_fsl=loss_fsl,
            start_prob=start_prob,
            end_prob=end_prob,
            attentions=attentions,
            fuzzy_span_attentions=fuzzy_span_attentions
        )
