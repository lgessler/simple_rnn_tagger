import torch

from rnn_tagger.data import read_data, make_data_loaders
from rnn_tagger.train import train_model
from rnn_tagger.transformer_model import TransformerTagger
from rnn_tagger.rnn_cell_model import RnnCellTagger
from rnn_tagger.lstm_model import LstmTagger


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    embedding_dim = 300
    hidden_dim = 512
    batch_size = 16
    token_vocab, tag_vocab, datasets = read_data()
    data_loaders = make_data_loaders(datasets, batch_size, device)

    model = TransformerTagger(hidden_dim, token_vocab, tag_vocab)
    model = LstmTagger(embedding_dim, hidden_dim, token_vocab, tag_vocab)
    model = RnnCellTagger(embedding_dim, hidden_dim, token_vocab, tag_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    train_model(model, data_loaders, optimizer, device)


if __name__ == '__main__':
    main()