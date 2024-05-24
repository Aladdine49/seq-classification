#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from collections import Counter
from pathlib import Path
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 
#print(torch.cuda.current_device())

path1 = "/home/alekchiri/files/Homo_sapiens.GRCh38.cdna_mod.fa"
path2 = "/home/alekchiri/files/lncipedia_5_2.fasta"
path3 = "/home/alekchiri/files/humrep.fa"
path_test = "/home/aladdine_lekchiri/Téléchargements/DB/fasta_test.fa"
files = [path1,path2,path3]
#path1 = "/home/aladdine_lekchiri/Téléchargements/DB/Homo_sapiens.GRCh38.cdna_mod.fa"
#path2 = "/home/aladdine_lekchiri/Téléchargements/DB/lncipedia_5_2.fasta"
#path3 = "/home/aladdine_lekchiri/Téléchargements/DB/humrep.fa"
#path_test = "/home/aladdine_lekchiri/Téléchargements/DB/fasta_test.fa"
#files = [path1,path2,path3]
# %%
def seq_finder(file):
    """
    Parses a FASTA file and extracts sequences with their IDs.

    This function reads a FASTA file and extracts the sequence IDs and the
    corresponding nucleotide sequences as strings. The sequences are returned 
    as a list of tuples, where each tuple contains a sequence ID and the 
    sequence itself.

    Parameters:
    file (str): The path to the input FASTA file.

    Returns:
    List[Tuple[str, str]]: A list of tuples where each tuple contains a 
                           sequence ID and its corresponding sequence.
    """
    sequences = list()
    for sequence in SeqIO.parse(file, "fasta"):
        sequences.append((sequence.id, str(sequence.seq)))
    return  sequences
# sequences = seq_finder(path_test)
# print(sequences[0:10])

def one_hot(seq):
    
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    seq_encoded = list()
    for nuc in seq.upper():
        if nuc in mapping:
            seq_encoded.append(mapping[nuc])
        else:
            seq_encoded.append(mapping['N'])
    return seq_encoded

# %%
#fragmentation maxseq = 70
seq = "AGTCCGATAGCTACGATCGATCGATCGATCGATACGATCGATCGATCGATCGATCGATCGATCGAAGCTATCGATCGTAGCfTAGCTGACTGACTGACTGATGCTAGCTAGCTAGCTAGCTAGCTAGCTGATCGATCGATCGACTGACTGACTGATG"

def fragmentor(sequence, id='', maxseq=75, overlap=10, max_gap=5):
    """
    Splits a sequence into fragments with the specified overlap and size.
    
    This function only fragments sequences longer than maxseq.

    Parameters:
    sequence (str): The input sequence to fragment.
    id (str): Identifier to associate with each fragment.
    maxseq (int): Maximum length of each fragment.
    overlap (int): Overlap between consecutive fragments.
    max_gap (int): Maximum percentage of gap '-' allowed in a fragment.

    Returns:
    list: A list of tuples where each tuple contains a fragment and its identifier.
    """
    fragments = []
    step_size = maxseq - overlap
    if len(sequence) <= maxseq:
        return fragments
    else:
        for i in range(0, len(sequence), step_size):
            fragment = sequence[i:i + maxseq]
            if len(fragment) < maxseq:
                fragment += 'N' * (maxseq - len(fragment))
            num_ = fragment.count('N')
            if ( num_ * 100 / maxseq) < max_gap:
                fragments.append((fragment,id))
    return fragments

# fragments = fragmentor(seq)
# for fragment in fragments:
#     print(fragment)


# %%
def preparing_seq(file, maxseq):
    """
    Prepares and processes sequences from a given file.

    This function reads sequences from a file, fragments longer sequences 
    using the `fragmentor` function, and pads shorter sequences to ensure
    uniform sequence length. It also labels the sequences based on the 
    prefix of the sequence ID.

    Parameters:
    file (str): The path to the file containing sequences in FASTA format.
    maxseq (int): The maximum sequence length. Sequences longer than this 
                  value will be fragmented, and shorter ones will be padded.

    Returns:
    tuple: A tuple containing:
        - res (list): A list of tuples where each tuple contains a sequence 
                      and its associated ID.
        - labels (list): A list of labels assigned to each sequence based 
                         on its ID prefix. The labels are:
                             - "protein_coding": for IDs starting with "ENS"
                             - "lncRNA": for IDs starting with "LIN"
                             - "rep": for all other IDs

    Raises:
    ValueError: If the lengths of `res` and `labels` lists are not the same.
    """
    res = list()
    labels = list()
    sequences = seq_finder(file)
    for id, sequence in sequences:
        if len(sequence) > maxseq:
            fragments = fragmentor(sequence,id=id,maxseq=maxseq)
            res.extend(fragments)
        elif len(sequence) < maxseq:
            padded_sequence = sequence + "N" * (maxseq - len(sequence))
            res.append((padded_sequence, id))
        else:
            res.append((sequence, id))
    for seq in res:
        if seq[1].startswith("ENS"):
            labels.append("protein_coding")
        elif seq[1].startswith("LIN"):
            labels.append("lncRNA")
        else:
            labels.append("rep")
    if len(res) != len(labels):
        raise ValueError("len res and len labels are not the same")
    return res, labels
path_test = "/home/aladdine_lekchiri/Téléchargements/DB/fasta_test.fa"

all_sequences = list()
all_labels = list()
sequence_ens,labels_ens = preparing_seq(path1,100)
sequence_lnc,labels_lnc = preparing_seq(path2,100)
sequence_rep,labels_rep = preparing_seq(path3,100)
print("Pre_processing done !")

# %%
all_sequences = sequence_ens + sequence_lnc + sequence_rep
all_labels = labels_ens + labels_lnc + labels_rep
seq_one_hot_code = list()
for seq, id in all_sequences:
    encoded = one_hot(seq)
    seq_one_hot_code.append(encoded)

# %%

def convert(labels):
    dico = {'protein_coding': 0, 'lncRNA' : 1, 'rep' : 2}
    num_labels = list()
    for lab in labels:
        num_labels.append(dico[lab])
    return num_labels



# %%
labels = convert(all_labels)
print(f'labels_number {len(all_labels)}, sequences_number {len(all_sequences)}')

# ### Implementing CNN model 

# class RNASequenceClassifier(nn.Module):
#     def __init__(self):
#         super(RNASequenceClassifier, self).__init__()

#         # Update convolution layers' parameters
#         self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=7)
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # Update dropout layer's parameters
#         self.dropout1 = nn.Dropout(0.1)

#         # Update dense layers' parameters
#         self.fc1 = nn.Linear(256 * 34, 256)  # Adjusted according to the new conv layer output
#         self.fc2 = nn.Linear(256, 3)

#     def forward(self, x):
#         # Apply convolutional layers and non-linearities
#         x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
#         x = self.pool(x)
#         x = self.dropout1(x)

#         # Flatten the data for the fully connected layer
#         x = torch.flatten(x, 1)  # Flatten from dimension 1

#         # Pass through fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = self.fc2(x)

#         return x
class RNASequenceClassifier(nn.Module):
    def __init__(self):
        super(RNASequenceClassifier, self).__init__()

        # Define convolution layers with different kernel sizes in independent models
        self.conv1_model = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=256, kernel_size=10),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )

        self.conv2_model = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=256, kernel_size=20),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )

        self.conv3_model = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=256, kernel_size=30),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )

        # Compute the output size after convolutions and pooling
        conv1_output_size = ((100 - 10 + 1) // 2)
        conv2_output_size = ((100 - 20 + 1) // 2)
        conv3_output_size = ((100 - 30 + 1) // 2)        
        self.fc_input_size = 256 * (conv1_output_size + conv2_output_size + conv3_output_size)
#self.fc_input_size = 256 * (32 + 33 + 30)  # Adjust according to the output size of the convolutions

        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        # Apply each independent convolutional model
        x1 = self.conv1_model(x)
        x2 = self.conv2_model(x)
        x3 = self.conv3_model(x)

        # Flatten the data for the fully connected layer
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x3 = torch.flatten(x3, 1)

        # Concatenate the outputs
        x = torch.cat((x1, x2, x3), dim=1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

# Assuming the rest of the code for data loading and training remains the same

# Load your data
data_tensor = torch.tensor(seq_one_hot_code, dtype=torch.float32).to(device)
labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
print(data_tensor.shape, labels_tensor.shape)

# Create dataset and dataloaders
dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
total_size = len(dataset)
train_size = int(0.7 * total_size)
validation_size = int(0.1 * total_size)
test_size = total_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=10, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=0)

# Calculate class weights for handling imbalanced datasets
class_counts = Counter(labels_tensor.tolist())
total_samples = len(labels_tensor)
class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Set preparation done!")

# Initialize model, optimizer, and loss function
model = RNASequenceClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

print("Running the model")
# Early stopping parameters
patience = 5  # Number of epochs to wait for improvement
best_val_accuracy = 0
epochs_no_improve = 0
# Training loop
model_path = Path('results_accuracy_100.txt')
model_path.write_text('')
losses = []
accuracies = []
val_losses = []
val_accuracies = []
with open(model_path, 'a') as fh:
    for epoch in range(50):
        epoch_losses = []
        epoch_accuracies = []
        model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 1)  # Ensure correct input dimensions for convolutions
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            epoch_accuracies.append(accuracy)

        epoch_loss = np.mean(epoch_losses)
        epoch_accuracy = np.mean(epoch_accuracies)
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            val_epoch_losses = []
            val_epoch_accuracies = []
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.permute(0, 2, 1)  # Ensure correct input dimensions for convolutions
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

                val_epoch_losses.append(val_loss.item())
                _, predicted = torch.max(outputs.data, 1)
                val_accuracy = (predicted == labels).sum().item() / labels.size(0)
                val_epoch_accuracies.append(val_accuracy)

            val_epoch_loss = np.mean(val_epoch_losses)
            val_epoch_accuracy = np.mean(val_epoch_accuracies)
            val_losses.append(val_epoch_loss)
            val_accuracies.append(val_epoch_accuracy)

        result = f'Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Val_Loss: {val_epoch_loss}, Val_Accuracy: {val_epoch_accuracy}\n'
        print(result.strip())
        fh.write(result)
         # Check for improvement
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
torch.save(model, '/home/alekchiri/model_3_kernels_100_after.pth')
