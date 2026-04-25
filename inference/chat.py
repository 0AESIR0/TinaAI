import torch, json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import PreTrainedTokenizerFast
from safetensors.torch import load_file
from model.config import ModelConfig
from model.model import RuyaGPT

MODEL_YOLU = "checkpoints/best"
AYGIT      = "cuda" if torch.cuda.is_available() else "cpu"

def yukle():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_YOLU)
    config    = ModelConfig.yukle(os.path.join(MODEL_YOLU, "config.json"))
    model     = RuyaGPT(config)
    durum     = load_file(os.path.join(MODEL_YOLU, "model.safetensors"), device=AYGIT)
    model.load_state_dict(durum)
    model.to(AYGIT).eval()
    return model, tokenizer

@torch.no_grad()
def uret(model, tokenizer, prompt, max_token=300, temp=0.8, top_p=0.92):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(AYGIT)
    eos = tokenizer.eos_token_id

    for _ in range(max_token):
        if ids.shape[1] >= model.config.max_position_embeddings:
            break
        logits = model(ids)["logits"][:, -1, :] / temp
        # Top-p
        probs = torch.softmax(logits, -1)
        sir   = torch.argsort(probs, descending=True)
        cum   = torch.cumsum(probs.gather(-1, sir), -1)
        kes   = cum - probs.gather(-1, sir) >= top_p
        logits[0][sir[0][kes[0]]] = float("-inf")
        probs  = torch.softmax(logits, -1)
        sonraki = torch.multinomial(probs, 1)
        ids = torch.cat([ids, sonraki], -1)
        if sonraki.item() == eos:
            break

    metin = tokenizer.decode(ids[0], skip_special_tokens=False)
    if "<|assistant|>" in metin:
        return metin.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
    return metin

def main():
    print("Model yükleniyor...")
    model, tokenizer = yukle()
    print(f"Hazır! ({sum(p.numel() for p in model.parameters())/1e6:.0f}M parametre)")
    print("Çıkmak için 'quit'\n")

    while True:
        kullanici = input("Sen: ").strip()
        if kullanici.lower() in ("quit", "exit", "çık", "q"):
            break
        prompt = f"<|user|>{kullanici}<|assistant|>"
        print(f"Bot: {uret(model, tokenizer, prompt)}\n")

if __name__ == "__main__":
    main()