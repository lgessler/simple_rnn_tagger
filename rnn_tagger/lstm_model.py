import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from allennlp_light.nn.util import sequence_cross_entropy_with_logits


class LstmTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, token_vocab, tag_vocab):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(len(token_vocab), embedding_dim, padding_idx=token_vocab.index("@@PAD@@"))
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.tag_head = nn.Linear(hidden_dim * 2, len(tag_vocab))
        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.accuracy = Accuracy(num_classes=len(self.tag_vocab), task="multiclass", top_k=1)

    def forward(self, tokens, tags=None):
        batch_size, seq_length = tokens.shape
        device = tokens.device
        embeds = self.word_embeddings(tokens)

        h_0 = (
            torch.zeros((2 * 2, batch_size, self.hidden_dim)).to(device),
            torch.zeros((2 * 2, batch_size, self.hidden_dim)).to(device)
        )
        stacked_states, _ = self.lstm(embeds, h_0)
        tag_logits = self.tag_head(stacked_states)

        mask = tokens.eq(0.0).logical_not()
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


