import rnn_tagger.data as data

token_vocab, tag_vocab, datasets = data.read_data()

datasets["train"][0]



for x in list(zip(*datasets["dev"].sentences[5])):
    print(x)

dev_data_loader = data_loaders["dev"]

first_batch = next(iter(dev_data_loader))

list(zip(*datasets["dev"].sentences[0]))

# I set/VERB the book on the table
# We finished the problem set/NOUN