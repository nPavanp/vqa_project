from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from langdetect import detect
from PIL import Image
import torch
import pickle
import torchvision.transforms as transforms

# ========================
# PERFORMANCE SETTINGS
# ========================
torch.set_num_threads(4)

# ========================
# DEVICE (CPU ONLY)
# ========================
device = torch.device("cpu")

# ========================
# LOAD BLIP2 (SAFE)
# ========================
print("Loading BLIP2...")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl"
)

blip_model.to(device)
blip_model.eval()

# ========================
# LOAD TRANSLATOR
# ========================
print("Loading Translator...")

translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

translator_model.to(device)
translator_model.eval()

lang_code_map = {
    "en":"eng_Latn","hi":"hin_Deva","te":"tel_Telu",
    "ta":"tam_Taml","kn":"kan_Knda","ml":"mal_Mlym"
}

def translate(text, src, tgt):
    translator_tokenizer.src_lang = lang_code_map[src]
    inputs = translator_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        tokens = translator_model.generate(
            **inputs,
            forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids(lang_code_map[tgt]),
            max_length=50
        )

    return translator_tokenizer.decode(tokens[0], skip_special_tokens=True)

# ========================
# LOAD CUSTOM MODEL
# ========================
from models.vqa_model import VQAModel

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

with open("weights/vocab.pkl","rb") as f:
    vocab = pickle.load(f)

with open("weights/answers.pkl","rb") as f:
    idx_to_answer = pickle.load(f)

custom_model = VQAModel(len(vocab),300,256,len(idx_to_answer))
custom_model.load_state_dict(torch.load("weights/vqa_model.pth", map_location=device))
custom_model.to(device)
custom_model.eval()

def encode_question(q):
    tokens = q.lower().split()
    enc = [vocab.get(w, vocab["<UNK>"]) for w in tokens]
    enc = enc[:20] + [vocab["<PAD>"]] * (20-len(enc))
    return torch.tensor(enc).unsqueeze(0)

# ========================
# CUSTOM MODEL
# ========================
def predict_custom_vqa(image_path, question):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    q = encode_question(question)

    with torch.no_grad():
        out = custom_model(image, q)
        _, pred = torch.max(out,1)

    return idx_to_answer[pred.item()]

# ========================
# BLIP2 (OPTIMIZED)
# ========================
def open_vqa(image_path, question):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, question, return_tensors="pt")

    with torch.no_grad():
        out = blip_model.generate(
            **inputs,
            max_new_tokens=15   # 🔥 reduced for speed
        )

    return processor.decode(out[0], skip_special_tokens=True)

# ========================
# FINAL PIPELINE
# ========================
def final_pipeline(image_path, question):
    lang = detect(question)

    if lang != "en":
        q_en = translate(question, lang, "en")
    else:
        q_en = question

    if "what is" in q_en.lower() or "this place" in q_en.lower():
        answer_en = open_vqa(image_path, q_en)
    else:
        answer_en = predict_custom_vqa(image_path, q_en)

    if lang != "en":
        return translate(answer_en, "en", lang)
    else:
        return answer_en

def predict(image_path, question):
    return final_pipeline(image_path, question)

# ========================
# WARMUP
# ========================
print("Warming up...")
dummy = Image.new("RGB", (224,224))
processor(dummy, "test", return_tensors="pt")

print("✅ Ready!")

# ========================
# TEST
# ========================
if __name__ == "__main__":
    print(predict("test.jpg","What is in the image?"))