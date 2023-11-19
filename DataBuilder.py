# ======================================================================================================================
# ----- DATA FORMAT -----
# sents_data_train --> texts for training (X)
# triplets_data_train --> corresponding triplets for training (Y)
# sents_data_validation --> texts for validation (X)
# triplets_data_validation --> corresponding triplets for validation (Y)
# sents_data_test --> texts for testing (X)
# triplets_data_test --> corresponding triplets for testing (Y)
# ======================================================================================================================

import pickle
from datasets import load_dataset
import gensim.downloader

# Load the datasets
dataset = load_dataset("docred")

triplet_token = "triplet"
subj_token = "subject"
obj_token = "object"

# Prepare the data
sents_data = {}
triplets_data = {}
for key in dataset.keys():
    sents_data[key] = []
    triplets_data[key] = []
    for row in dataset[key]:
        triplets = []
        for i in range(len(row['labels']['relation_text'])):
            triplets.append([triplet_token, row['vertexSet'][row['labels']['head'][i]][0]['name'], subj_token, row['vertexSet'][row['labels']['tail'][i]][0]['name'], obj_token, row['labels']['relation_text'][i]])
        sents_data[key].append(row['sents'])
        triplets_data[key].append(triplets)

# Concatenate the annotated and distant data
sents_data['train'] = sents_data['train_annotated'] + sents_data['train_distant']
triplets_data['train'] = triplets_data['train_annotated'] + triplets_data['train_distant']

# Convert the data to string format
sents_data['train'] = [' '.join([' '.join(sent) for sent in sents]) for sents in sents_data['train']]
sents_data['validation'] = [' '.join([' '.join(sent) for sent in sents]) for sents in sents_data['validation']]
sents_data['test'] = [' '.join([' '.join(sent) for sent in sents]) for sents in sents_data['test']]
triplets_data['train'] = [' '.join([' '.join(triplet) for triplet in triplets]) for triplets in triplets_data['train']]
triplets_data['validation'] = [' '.join([' '.join(triplet) for triplet in triplets]) for triplets in triplets_data['validation']]
triplets_data['test'] = [' '.join([' '.join(triplet) for triplet in triplets]) for triplets in triplets_data['test']]

# Pickle the lists
with open('data/triplets_data_train.pkl', 'wb') as f:
    pickle.dump(triplets_data['train'], f)

with open('data/triplets_data_validation.pkl', 'wb') as f:
    pickle.dump(triplets_data['validation'], f)

with open('data/triplets_data_test.pkl', 'wb') as f:
    pickle.dump(triplets_data['test'], f)

with open('data/sents_data_train.pkl', 'wb') as f:
    pickle.dump(sents_data['train'], f)

with open('data/sents_data_validation.pkl', 'wb') as f:
    pickle.dump(sents_data['validation'], f)

with open('data/sents_data_test.pkl', 'wb') as f:
    pickle.dump(sents_data['test'], f)

# Load pretrained "word2vec" embeddings
w2v = gensim.downloader.load('word2vec-google-news-300')

# Pickle the w2v
with open('data/w2v.pkl', 'wb') as f:
    pickle.dump(w2v, f)