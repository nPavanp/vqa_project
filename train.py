from datasets import load_dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from collections import Counter
import pickle
import re
from tqdm import tqdm
import os

# ========================
# CONFIG
# ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 32
LR = 5e-4
MAX_LEN = 20

# ========================
# LOAD DATASET
# ========================
dataset = load_dataset("flaviagiammarino/vqa-rad")
df = pd.DataFrame(dataset["train"])
df = df[["image", "question", "answer"]]

# ========================
# CLEAN TEXT
# ========================
def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9 ]", "", text)

df["question"] = df["question"].apply(clean_text)
df["answer"] = df["answer"].apply(clean_text)

# ========================
# FILTER TOP ANSWERS
# ========================
top_answers = df["answer"].value_counts().nlargest(50).index
df = df[df["answer"].isin(top_answers)]

answer_to_idx = {a:i for i,a in enumerate(top_answers)}
idx_to_answer = {i:a for a,i in answer_to_idx.items()}
df["answer_encoded"] = df["answer"].apply(lambda x: answer_to_idx[x])

# ========================
# VOCAB
# ========================
vocab = {"<PAD>":0, "<UNK>":1}
counter = Counter()

for q in df["question"]:
    for w in q.split():
        counter[w] += 1

idx = 2
for word, count in counter.items():
    if count > 2:
        vocab[word] = idx
        idx += 1

def encode_question(q):
    tokens = q.split()
    enc = [vocab.get(w, vocab["<UNK>"]) for w in tokens]
    enc = enc[:MAX_LEN] + [vocab["<PAD>"]] * (MAX_LEN - len(enc))
    return enc

df["question_encoded"] = df["question"].apply(encode_question)

# ========================
# DATASET CLASS
# ========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class VQADataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = row["image"].convert("RGB")
        image = transform(image)

        question = torch.tensor(row["question_encoded"])
        answer = torch.tensor(row["answer_encoded"])

        return image, question, answer

# ========================
# SPLIT DATA
# ========================
dataset_full = VQADataset(df)
train_size = int(0.8 * len(dataset_full))
val_size = len(dataset_full) - train_size

train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ========================
# MODEL
# ========================
import torchvision.models as models

class VQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_answers):
        super().__init__()

        self.cnn = models.resnet18(weights="DEFAULT")
        self.cnn.fc = nn.Identity()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(512 + hidden_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_answers)

    def forward(self, image, question):
        img_feat = self.cnn(image)

        q_embed = self.embedding(question)
        _, (h, _) = self.lstm(q_embed)
        q_feat = h.squeeze(0)

        x = self.relu(self.fc1(torch.cat((img_feat, q_feat), dim=1)))
        return self.fc2(x)

model = VQAModel(len(vocab), 300, 256, len(answer_to_idx)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ========================
# TRAIN LOOP
# ========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, questions, answers in tqdm(train_loader):
        images, questions, answers = images.to(DEVICE), questions.to(DEVICE), answers.to(DEVICE)

        outputs = model(images, questions)
        loss = criterion(outputs, answers)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, questions, answers in val_loader:
            images, questions, answers = images.to(DEVICE), questions.to(DEVICE), answers.to(DEVICE)

            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            val_loss += loss.item()

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")

# ========================
# SAVE MODEL
# ========================
os.makedirs("weights", exist_ok=True)

torch.save(model.state_dict(), "weights/vqa_model.pth")

with open("weights/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("weights/answers.pkl", "wb") as f:
    pickle.dump(idx_to_answer, f)

print("\n✅ Training Complete & Model Saved!")