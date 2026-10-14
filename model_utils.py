import torch
import torch.nn as nn
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
import string

# 1. Load the exported model state and vocab[cite: 5]
checkpoint = torch.load('classifier.pt', weights_only=False) #[cite: 5]
vocab = checkpoint['vocab']
label_map = checkpoint['label_map']

# Define the exact model architecture so PyTorch can map the weights
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=4, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed)
        h = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(h))

model = IntentClassifier(vocab_size=len(vocab))
model.load_state_dict(checkpoint['model_state']) #[cite: 5]
model.eval()

# 2. Classifier Prediction Function[cite: 7]
def classify_intent(text):
    clean_text = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    indices = [vocab.get(token, vocab["<unk>"]) for token in clean_text][:40]
    if not indices: indices = [vocab["<unk>"]]
    
    seq = torch.tensor([indices])
    length = torch.tensor([len(indices)])
    
    with torch.no_grad():
        output = model(seq, length)
        probs = torch.softmax(output, dim=1)[0]
        confidence, label_idx = torch.max(probs, dim=0)
        
    return label_map[label_idx.item()], confidence.item() #[cite: 5, 7]