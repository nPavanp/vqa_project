import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 20
EMBED_DIM = 300
HIDDEN_DIM = 256
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 5

MODEL_PATH = "weights/vqa_model.pth"
VOCAB_PATH = "weights/vocab.pkl"
ANSWER_PATH = "weights/answers.pkl"
