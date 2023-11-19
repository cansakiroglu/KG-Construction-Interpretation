import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

class CustomDataset(Dataset):
    def __init__(self, X, Y, tokenizer, max_seq_length_for_input=600, max_seq_length_for_target=1200):
        self.X = X
        self.Y = Y
        self.tokenizer = tokenizer
        self.max_seq_length_for_input = max_seq_length_for_input
        self.max_seq_length_for_target = max_seq_length_for_target

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        # Tokenize input and target sequences with truncation and padding
        input_ids = self.tokenizer.encode(x, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_seq_length_for_input)[0]
        target_ids = self.tokenizer.encode(y, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_seq_length_for_target)[0]

        return {'input_ids': input_ids, 'labels': target_ids}


# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

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

# Assuming 'sents' is your list of X and 'triplets' is your list of Y
dataset = CustomDataset(sents_data_train, triplets_data_train, tokenizer)

# Define hyperparameters
batch_size = 4
learning_rate = 3e-5
epochs = 10

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # ?: num_workers

# Prepare optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
device = torch.device("mps")
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_bart")
tokenizer.save_pretrained("fine_tuned_bart")