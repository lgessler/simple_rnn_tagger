import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from allennlp_light.nn.util import sequence_cross_entropy_with_logits
from allennlp_light.modules.seq2seq_encoders import PytorchTransformer


class TransformerTagger(nn.Module):

    def __init__(self, hidden_dim, token_vocab, tag_vocab):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(len(token_vocab), hidden_dim, padding_idx=token_vocab.index("@@PAD@@"))
        self.transformer = PytorchTransformer(
            hidden_dim,
            1,
            feedforward_hidden_dim=int(hidden_dim * 2.5),
            positional_encoding="sinusoidal"
        )
        self.tag_head = nn.Linear(hidden_dim, len(tag_vocab))
        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.accuracy = Accuracy(num_classes=len(self.tag_vocab), task="multiclass", top_k=1)

    def forward(self, tokens, tags=None):
        mask = tokens.eq(0.0).logical_not()
        embeds = self.word_embeddings(tokens)
        hidden_states = self.transformer(embeds, mask)
        tag_logits = self.tag_head(hidden_states)

        class_probs = F.softmax(tag_logits)
        pred_tags = class_probs.argmax(-1)

        output = {"logits": tag_logits, "preds": pred_tags * mask}
        if tags is not None:
            loss = sequence_cross_entropy_with_logits(tag_logits, tags, mask, average="token")
            output["loss"] = loss
            flat_preds = pred_tags.masked_select(mask)
            flat_tags = tags.masked_select(mask)
            acc = self.accuracy(flat_preds, flat_tags)
            num_words_correct = (flat_preds == flat_tags).eq(True).sum(-1).item()
            num_words = len(flat_preds)
            output["accuracy"] = acc
            output["num_words_correct"] = num_words_correct
            output["num_words"] = num_words
            self.accuracy.reset()

        return output


