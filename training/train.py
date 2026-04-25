import argparse
import json
import os
import random
import sys
from contextlib import nullcontext

import torch
from datasets import DatasetDict, load_from_disk
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import ModelConfig
from model.model import RuyaGPT


def parse_args():
    parser = argparse.ArgumentParser(description="Train RuyaGPT locally or on Kaggle/Colab.")
    parser.add_argument("--data-dir", default="data/cleaned_tr")
    parser.add_argument("--tokenizer-dir", default="tokenizer/tr_tokenizer")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=300_000)
    parser.add_argument("--valid-samples", type=int, default=5_000)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--disable-grad-checkpoint", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_amp_context_factory(device: torch.device):
    if device.type != "cuda":
        return nullcontext, False

    if torch.cuda.is_bf16_supported():
        return lambda: torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), True
    return lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16), True


class TurkceDataset(Dataset):
    def __init__(self, ds, tokenizer, max_len, max_ornek=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ornekler = []

        hedef = len(ds) if max_ornek is None else min(max_ornek, len(ds))
        print(f"Dataset hazirlaniyor ({hedef:,} ornek)...")

        for i in tqdm(range(hedef), desc="Tokenize filtreleme"):
            metin = ds[i]["text"]
            if not metin or len(metin) < 50:
                continue
            self.ornekler.append(metin)

        print(f"Kullanilabilir ornek: {len(self.ornekler):,}")

    def __len__(self):
        return len(self.ornekler)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.ornekler[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return input_ids, attention_mask, labels


def save_checkpoint(model, config, tokenizer, yol, optimizer=None, scheduler=None, scaler=None, trainer_state=None):
    os.makedirs(yol, exist_ok=True)
    kaydedilecek_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    tensors = {k: v.contiguous().cpu() for k, v in kaydedilecek_model.state_dict().items()}
    save_file(tensors, os.path.join(yol, "model.safetensors"))
    config.kaydet(os.path.join(yol, "config.json"))
    tokenizer.save_pretrained(yol)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(yol, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(yol, "scheduler.pt"))
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(yol, "scaler.pt"))
    if trainer_state is not None:
        with open(os.path.join(yol, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, indent=2, ensure_ascii=False)

    print(f"  Checkpoint kaydedildi -> {yol}")


def load_resume_state(resume_dir, model, optimizer, scheduler, scaler, device):
    model_path = os.path.join(resume_dir, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Resume klasorunde model.safetensors bulunamadi: {resume_dir}")

    durum = load_file(model_path, device=str(device))
    model.load_state_dict(durum)

    state = {
        "epoch": 0,
        "global_step": 0,
        "best_val_loss": float("inf"),
    }

    trainer_state_path = os.path.join(resume_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, encoding="utf-8") as f:
            state.update(json.load(f))

    optimizer_path = os.path.join(resume_dir, "optimizer.pt")
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    scheduler_path = os.path.join(resume_dir, "scheduler.pt")
    if os.path.exists(scheduler_path):
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))

    scaler_path = os.path.join(resume_dir, "scaler.pt")
    if os.path.exists(scaler_path):
        scaler.load_state_dict(torch.load(scaler_path, map_location=device))

    return state


def validate(model, dl, device, amp_context_factory):
    model.eval()
    toplam, sayi = 0.0, 0
    with torch.no_grad():
        for input_ids, attn_mask, labels in dl:
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp_context_factory():
                cikti = model(input_ids, attn_mask, labels)

            toplam += cikti["loss"].item()
            sayi += 1
            if sayi >= 100:
                break

    model.train()
    return toplam / max(sayi, 1)


def build_dataloaders(args, tokenizer, device):
    ds = load_from_disk(args.data_dir)
    if isinstance(ds, DatasetDict):
        if "train" not in ds or "test" not in ds:
            raise ValueError("DatasetDict bulundu ama train/test split yok.")
        split_ds = ds
    else:
        split_ds = ds.train_test_split(test_size=args.valid_ratio, seed=args.seed)

    train_ds = TurkceDataset(split_ds["train"], tokenizer, args.max_len, args.max_samples)
    valid_ds = TurkceDataset(split_ds["test"], tokenizer, args.max_len, args.valid_samples)

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_dl = DataLoader(train_ds, shuffle=True, **loader_args)
    valid_dl = DataLoader(valid_ds, shuffle=False, **loader_args)
    return train_ds, valid_ds, train_dl, valid_dl


def egit():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = detect_device()
    amp_context_factory, use_amp = get_amp_context_factory(device)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"Aygit: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    tokenizer.pad_token = tokenizer.pad_token or "<|pad|>"
    tokenizer.bos_token = tokenizer.bos_token or "<|bos|>"
    tokenizer.eos_token = tokenizer.eos_token or "<|eos|>"
    tokenizer.padding_side = "right"

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_len,
        hidden_dropout_prob=args.dropout,
        attention_dropout_prob=args.dropout,
    )
    model = RuyaGPT(config).to(device)
    temel_model = model

    if not args.disable_grad_checkpoint:
        model.enable_gradient_checkpointing()

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    print(f"\nModel: {temel_model.param_sayisi() / 1e6:.1f}M parametre")
    print(f"Gradient checkpointing: {'acik' if not args.disable_grad_checkpoint else 'kapali'}")

    train_ds, valid_ds, train_dl, valid_dl = build_dataloaders(args, tokenizer, device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    toplam_adim = max(1, ((len(train_dl) + args.grad_accum - 1) // args.grad_accum) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, toplam_adim),
        num_training_steps=toplam_adim,
    )

    baslangic_epoch = 0
    global_adim = 0
    en_iyi_loss = float("inf")

    if args.resume_from:
        state = load_resume_state(args.resume_from, model, optimizer, scheduler, scaler, device)
        baslangic_epoch = int(state.get("epoch", 0))
        global_adim = int(state.get("global_step", 0))
        en_iyi_loss = float(state.get("best_val_loss", float("inf")))
        print(f"Resume aktif -> {args.resume_from} (epoch={baslangic_epoch}, step={global_adim})")

    print(f"Train: {len(train_ds):,} | Valid: {len(valid_ds):,}")
    print(f"Toplam optimizer adimi: {toplam_adim:,}")
    print("\nEgitim basliyor...\n")

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(baslangic_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        mikro_adim = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for adim, (input_ids, attn_mask, labels) in enumerate(pbar):
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp_context_factory():
                cikti = model(input_ids, attn_mask, labels)
                loss = cikti["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum
            mikro_adim += 1

            step_due = mikro_adim == args.grad_accum or adim == len(train_dl) - 1
            if not step_due:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            mikro_adim = 0
            global_adim += 1
            ort_loss = epoch_loss / (adim + 1)
            pbar.set_postfix(
                {
                    "loss": f"{ort_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": global_adim,
                }
            )

            if global_adim % args.save_every == 0:
                v_loss = validate(model, valid_dl, device, amp_context_factory)
                print(f"\n[Step {global_adim}] Train: {ort_loss:.4f} | Valid: {v_loss:.4f}")
                save_checkpoint(
                    model,
                    config,
                    tokenizer,
                    os.path.join(args.output_dir, f"step-{global_adim}"),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    trainer_state={
                        "epoch": epoch,
                        "global_step": global_adim,
                        "best_val_loss": en_iyi_loss,
                    },
                )

        v_loss = validate(model, valid_dl, device, amp_context_factory)
        ort = epoch_loss / max(len(train_dl), 1)
        print(f"\nEpoch {epoch + 1} -> Train Loss: {ort:.4f} | Valid Loss: {v_loss:.4f}")

        if v_loss < en_iyi_loss:
            en_iyi_loss = v_loss
            save_checkpoint(
                model,
                config,
                tokenizer,
                os.path.join(args.output_dir, "best"),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                trainer_state={
                    "epoch": epoch + 1,
                    "global_step": global_adim,
                    "best_val_loss": en_iyi_loss,
                },
            )
            print("  En iyi model guncellendi.")

    save_checkpoint(
        model,
        config,
        tokenizer,
        os.path.join(args.output_dir, "final"),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer_state={
            "epoch": args.epochs,
            "global_step": global_adim,
            "best_val_loss": en_iyi_loss,
        },
    )
    print("\nEgitim tamamlandi.")


if __name__ == "__main__":
    egit()
