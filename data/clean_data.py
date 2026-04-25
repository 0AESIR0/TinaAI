import argparse
import os
import re

from datasets import Dataset, load_from_disk
from tqdm import tqdm


MIN_UZUNLUK = 100
MAX_UZUNLUK = 4096


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Turkish text dataset.")
    parser.add_argument("--input-dir", default="data/combined_tr")
    parser.add_argument("--output-dir", default="data/cleaned_tr")
    parser.add_argument("--min-length", type=int, default=MIN_UZUNLUK)
    parser.add_argument("--max-length", type=int, default=MAX_UZUNLUK)
    return parser.parse_args()


def metni_temizle(metin: str, min_uzunluk: int, max_uzunluk: int) -> str | None:
    if not metin or not isinstance(metin, str):
        return None

    if len(metin) < min_uzunluk or len(metin) > max_uzunluk:
        return None

    metin = re.sub(r"<[^>]+>", " ", metin)
    metin = re.sub(r"http\S+|www\.\S+", "", metin)
    metin = re.sub(r"\s+", " ", metin)
    metin = metin.strip()

    turkce = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")
    harf_sayisi = sum(1 for c in metin if c.isalpha())
    if harf_sayisi < 50:
        return None

    turkce_oran = sum(1 for c in metin if c in turkce) / harf_sayisi
    if turkce_oran < 0.7:
        return None

    return metin


def temizle():
    args = parse_args()
    print("Dataset yukleniyor...")
    ds = load_from_disk(args.input_dir)
    print(f"Ham veri: {len(ds):,} ornek")

    temiz_metinler = []
    atlanan = 0

    print("Temizleniyor...")
    for ornek in tqdm(ds, desc="Temizleme"):
        temiz = metni_temizle(ornek.get("text", ""), args.min_length, args.max_length)
        if temiz:
            temiz_metinler.append({"text": temiz})
        else:
            atlanan += 1

    print(f"\nTemiz: {len(temiz_metinler):,} | Atlandi: {atlanan:,}")

    temiz_ds = Dataset.from_list(temiz_metinler)
    os.makedirs(os.path.dirname(args.output_dir) or ".", exist_ok=True)
    temiz_ds.save_to_disk(args.output_dir)
    print(f"Kaydedildi -> {args.output_dir}")


if __name__ == "__main__":
    temizle()
