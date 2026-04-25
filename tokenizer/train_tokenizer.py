import argparse
import json
import os

from datasets import load_from_disk
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing


OZEL_TOKENLER = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train tokenizer for RuyaGPT.")
    parser.add_argument("--data-dir", default="data/cleaned_tr")
    parser.add_argument("--output-dir", default="tokenizer/tr_tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--sample-count", type=int, default=200_000)
    parser.add_argument("--min-frequency", type=int, default=3)
    parser.add_argument("--model-max-length", type=int, default=1024)
    return parser.parse_args()


def metin_akisi(ds, n):
    for i, ornek in enumerate(ds):
        if i >= n:
            break
        yield ornek["text"]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Dataset yukleniyor...")
    ds = load_from_disk(args.data_dir)
    print(f"Toplam ornek: {len(ds):,}")

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=OZEL_TOKENLER,
        min_frequency=args.min_frequency,
        show_progress=True,
    )

    print(f"\nTokenizer egitiliyor ({args.sample_count:,} ornek)...")
    tokenizer.train_from_iterator(metin_akisi(ds, args.sample_count), trainer=trainer)

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|>:0 $A:0 <|eos|>:0",
        special_tokens=[("<|bos|>", bos_id), ("<|eos|>", eos_id)],
    )

    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "model_max_length": args.model_max_length,
    }
    with open(os.path.join(args.output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    enc = tokenizer.encode("Merhaba, nasilsin?")
    print("\nTokenizer hazir")
    print(f"Vocab: {tokenizer.get_vocab_size():,}")
    print(f"Test: 'Merhaba, nasilsin?' -> {enc.tokens}")
    print(f"Kaydedildi -> {args.output_dir}")


if __name__ == "__main__":
    main()
