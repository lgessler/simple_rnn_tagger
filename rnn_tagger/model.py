import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from allennlp_light.nn.util import sequence_cross_entropy_with_logits


class RnnTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, token_vocab, tag_vocab):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(len(token_vocab), embedding_dim, padding_idx=token_vocab.index("@@PAD@@"))
        self.rnn_cell = nn.GRUCell(embedding_dim, hidden_dim)
        self.tag_head = nn.Linear(hidden_dim, len(tag_vocab))
        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.accuracy = Accuracy(num_classes=len(self.tag_vocab), task="multiclass", top_k=1)

    def forward(self, tokens, tags=None):
        batch_size, seq_length = tokens.shape
        device = tokens.device
        embeds = self.word_embeddings(tokens)

        h_n = None
        states = []
        for i in range(seq_length):
            h_n = self.rnn_cell(embeds[:, i], h_n)
            states.append(h_n)

        stacked_states = torch.stack(states, dim=1).to(device)
        tag_logits = self.tag_head(stacked_states)

        mask = tokens.eq(0.0).logical_not()
        class_probs = F.softmax(tag_logits)
        pred_tags = class_probs.argmax(-1)

        output = {"logits": tag_logits, "preds": pred_tags * mask}
        if tags is not None:
            loss = sequence_cross_entropy_with_logits(stacked_states, tags, mask, average="token")
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


