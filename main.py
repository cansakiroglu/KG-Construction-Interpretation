import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import gensim.downloader
import pickle

class CustomDataset(Dataset):
    def __init__(self, X, Y, w2v_model, max_input_length, max_target_length):
        self.X = X
        self.Y = Y
        self.w2v_model = w2v_model

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = [self.w2v_model[word] for word in self.X[idx].split() if word in w2v.index_to_key]
        x = x + [np.zeros(w2v.vector_size)] * (max_input_length - len(x))

        y = [self.w2v_model[word] for word in self.Y[idx].split() if word in w2v.index_to_key]
        y = y + [np.zeros(w2v.vector_size)] * (max_target_length - len(y))

        return {'input': torch.FloatTensor(np.array(x)), 'target': torch.FloatTensor(np.array(y))}

# Define the Encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        # Define the GRU layer
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, word_embedding, prev_hidden):
        output, hidden = self.gru(word_embedding, prev_hidden)
        return output, hidden

# Define the Decoder model
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()

        # Define the GRU layer
        self.gru = nn.GRU(input_size, hidden_size)

        # Define the output layer and the softmax activation function
        self.out = nn.Linear(hidden_size, len(w2v.index_to_key))
        self.softmax = nn.LogSoftmax()

    def forward(self, word_embedding, prev_hidden):
        output, hidden = self.gru(word_embedding, prev_hidden)
        output = self.softmax(self.out(output))
        return output, hidden

# Define the Seq2Seq model that combines Encoder and Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.prev_hidden = torch.zeros(1, 600, hidden_size).to(device)

    def forward(self, input_seq):
        # Encoder forward pass
        encoder_output, encoder_hidden = self.encoder(input_seq, self.prev_hidden)

        self.prev_hidden = encoder_hidden

        decoder_output, decoder_hidden = self.decoder(input_seq, encoder_hidden)

        return decoder_output

if __name__ == '__main__':

    print('Loading the pretrained "word2vec" embeddings...')

    # Load pretrained "word2vec" embeddings
    with open('data/w2v.pkl', 'rb') as f:
        w2v = pickle.load(f)

    print('Done!')

    print('Loading the data...')

    # Load the data
    with open('data/triplets_data_train.pkl', 'rb') as f:
        triplets_data_train = pickle.load(f)
    with open('data/triplets_data_validation.pkl', 'rb') as f:
        triplets_data_validation = pickle.load(f)
    with open('data/triplets_data_test.pkl', 'rb') as f:
        triplets_data_test = pickle.load(f)
    with open('data/sents_data_train.pkl', 'rb') as f:
        sents_data_train = pickle.load(f)
    with open('data/sents_data_validation.pkl', 'rb') as f:
        sents_data_validation = pickle.load(f)
    with open('data/sents_data_test.pkl', 'rb') as f:
        sents_data_test = pickle.load(f)

    # Create custom dataset
    max_input_length = 600 # max(max([len(x.split()) for x in sents_data_train]), max([len(x.split()) for x in sents_data_validation]), max([len(x.split()) for x in sents_data_test]))
    max_target_length = 1200 # max(max([len(x.split()) for x in triplets_data_train]), max([len(x.split()) for x in triplets_data_validation]), max([len(x.split()) for x in triplets_data_test]))
    train_data = CustomDataset(sents_data_train, triplets_data_train, w2v, max_input_length, max_target_length)

    # Define a batch size
    batch_size = 4

    # Create a DataLoader
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    print('Done!')

    print('Setting the device...')

    # Set the device
    # (MPS: Metal Performance Shaders) --> (Apple's API for GPU acceleration)     !!! Change with proper device if needed !!!
    device = torch.device("mps")

    print('Done!')

    # Define hyperparameters
    hidden_size = 256
    learning_rate = 0.001
    epochs = 10

    print('Initializing models and optimizer...')

    # Initialize models and optimizer
    input_size = w2v.vector_size
    encoder = Encoder(input_size, hidden_size).to(device)
    decoder = Decoder(input_size, hidden_size).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print('Done!')

    print('Starting the training process...')

    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            input_seq, target_seq = batch['input'].to(device), batch['target'].to(device)
            optimizer.zero_grad()
            outputs = model(input_seq)
            loss = criterion(outputs.view(-1, w2v.vector_size), target_seq.view(-1))
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}')