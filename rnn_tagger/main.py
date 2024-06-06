import torch

from rnn_tagger.data import read_data, make_data_loaders
from rnn_tagger.model import RnnTagger

from rnn_tagger.train import train_model


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    embedding_dim = 300
    hidden_dim = 1024
    batch_size = 128
    token_vocab, tag_vocab, datasets = read_data()
    data_loaders = make_data_loaders(datasets, batch_size, device)

    model = RnnTagger(embedding_dim, hidden_dim, token_vocab, tag_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, data_loaders, optimizer, device)


if __name__ == '__main__':
    main()