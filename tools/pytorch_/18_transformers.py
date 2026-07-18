import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn

vocab_size = 27
char_to_code = {"$": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26}
code_to_char = {0: "$", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h", 9: "i", 10: "j", 11: "k", 12: "l", 13: "m", 14: "n", 15: "o", 16: "p", 17: "q", 18: "r", 19: "s", 20: "t", 21: "u", 22: "v", 23: "w", 24: "x", 25: "y", 26: "z"}


names = []
with open('resources/names_2022.txt', 'r') as file:
    for line in file:
        name, _, _= line.lower().strip().split(',')
        names.append("$" + name + "$")

# converts from characters to integers and vice-versa
encode = lambda word: torch.tensor([char_to_code[c] for c in word])
decode = lambda tensor_i: ''.join(code_to_char[i.item()] for i in tensor_i)

example_name = "$ada$"
print(encode(example_name))
print(decode(encode(example_name)))

name_indices = [encode(name) for name in names]
target_indices = [name_index[1:] for name_index in name_indices]

X = pad_sequence(name_indices, batch_first=True, padding_value=0)
max_name_length = max(len(name) for name in names)
target_indices.append(torch.empty((max_name_length), dtype=torch.long))
Y = pad_sequence(target_indices, batch_first=True, padding_value=-1)[:-1]
print(X[0])
print(Y[0])

def get_batch(batch_size=64):
    random_idx = torch.randint(0, X.size(0), (batch_size,))
    inputs = X[random_idx]
    labels = Y[random_idx]
    return inputs, labels

inputs, labels = get_batch(3)
print(inputs)
print(labels)

embedding_dim = 3
embedding = nn.Embedding(vocab_size, embedding_dim)
example_input = torch.tensor([1, 1, 0, 2])
input_embd = embedding(example_input)
print(input_embd.shape)
print(input_embd)

class SequenceMLP(nn.Module):
    def __init__(self, vocab_size, max_sequence_length, embedding_dim, hidden_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim * max_sequence_length, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        sequence_embeddings = torch.zeros(batch_size, seq_len, self.max_sequence_length * self.embedding_dim)
        for i in range(seq_len):
            subsequence = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.int)
            prefix = x[:, :i+1]
            subsequence[:, :i+1] = prefix
            emb = self.embedding(subsequence)
            sequence_embeddings[:, i, :] = emb.view(batch_size, -1)

        x = self.linear(sequence_embeddings)
        x = self.relu(x)
        x = self.out(x)
        return x

embedding_dim = 3
max_sequence_length = X.shape[1]
model = SequenceMLP(vocab_size, max_sequence_length, embedding_dim)


import torch.optim as optim

def train(model, optimizer, num_steps=10_001, loss_report_interval=1_000):
    losses = []
    for i in range(1, num_steps):
        inputs, labels = get_batch()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-1)
        losses.append(loss.item())
        if i % loss_report_interval == 0:
        print(f'Average loss at step {i}: {sum(losses[loss_report_interval:]) / loss_report_interval:.4f}')
        loss.backward()
        optimizer.step()
optimizer = optim.SGD(model.parameters(), lr=0.1)

train(model, optimizer)

def generate_samples(model, num_samples=1, max_len=max_name_length):
    sequences = torch.zeros((num_samples, 1)).int()
    for _ in range(max_len):
        logits = model(sequences)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        sequences = torch.cat((sequences, idx_next), dim=1)
        for sequence in sequences:
        indices = torch.where(sequence == 0)[0]
        end = indices[1] if len(indices) > 1 else max_len
        sequence = sequence[1:end]
        print(decode(sequence))

generate_samples(model, num_samples=10)