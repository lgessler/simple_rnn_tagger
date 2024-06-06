import torch
import conllu
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class POSDataset(Dataset):
    def __init__(self, sentences, token_vocab, tag_vocab):
        self.sentences = sentences
        self.token_vocab = {t: i for i, t in enumerate(token_vocab)}
        self.tag_vocab = {t: i for i, t in enumerate(tag_vocab)}

    def __getitem__(self, idx):
        tokens, tags = self.sentences[idx]
        tokens = [self.token_vocab[t] if t in self.token_vocab else self.token_vocab["@@UNK@@"] for t in tokens]
        tags = [self.tag_vocab[t] for t in tags]
        item = {
            "tokens": torch.tensor(tokens),
            "tags": torch.tensor(tags),
        }
        return item

    def __len__(self):
        return len(self.sentences)


def read_file(path):
    with open(path, 'r') as f:
        sentences = conllu.parse(f.read())
    new_sentences = []
    for sentence in sentences:
        tokens = [t["form"] for t in sentence if isinstance(t["id"], int)]
        tags = [t["upos"] for t in sentence if isinstance(t["id"], int)]
        assert len(tokens) == len(tags)
        new_sentences.append((tokens, tags))
    return new_sentences


def read_splits():
    return {
        "train": read_file("data/en_gum-ud-train.conllu"),
        "dev": read_file("data/en_gum-ud-dev.conllu"),
        "test": read_file("data/en_gum-ud-test.conllu"),
    }


def make_vocab(raw_data):
    token_set = set()
    tag_set = set()

    for tokens, tags in raw_data["train"]:
        for token, tag in zip(tokens, tags):
            token_set.add(token)
            tag_set.add(tag)

    token_vocab = ["@@PAD@@", "@@UNK@@"] + sorted(list(token_set))
    tag_vocab = ["@@PAD@@"] + sorted(list(tag_set))
    return token_vocab, tag_vocab


def read_data():
    raw_data = read_splits()
    token_vocab, tag_vocab = make_vocab(raw_data)
    datasets = {
        "train": POSDataset(raw_data["train"], token_vocab, tag_vocab),
        "dev": POSDataset(raw_data["dev"], token_vocab, tag_vocab),
        "test": POSDataset(raw_data["test"], token_vocab, tag_vocab),
    }
    return token_vocab, tag_vocab, datasets


def make_collate(device):
    def collate(batch):
        output = {}
        output["tokens"] = pad_sequence([item["tokens"] for item in batch], batch_first=True).to(device)
        output["tags"] = pad_sequence([item["tags"] for item in batch], batch_first=True).to(device)
        return output
    return collate


def make_data_loaders(datasets, batch_size, device):
    collate_fn = make_collate(device)
    train_data_loader = DataLoader(datasets["train"], batch_size, shuffle=True, collate_fn=collate_fn)
    dev_data_loader = DataLoader(datasets["dev"], batch_size, shuffle=False, collate_fn=collate_fn)
    test_data_loader = DataLoader(datasets["test"], batch_size, shuffle=False, collate_fn=collate_fn)
    return {
        "train": train_data_loader,
        "dev": dev_data_loader,
        "test": test_data_loader,
    }

