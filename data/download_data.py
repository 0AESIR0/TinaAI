import argparse
import os

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare Turkish text datasets.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--culturax-limit", type=int, default=300_000)
    parser.add_argument("--opus-limit", type=int, default=100_000)
    return parser.parse_args()


def kaydet_ve_don(isim, ornekler, cikti_klasoru):
    yol = os.path.join(cikti_klasoru, f"{isim}_ds")
    ds = Dataset.from_list(ornekler)
    ds.save_to_disk(yol)
    print(f"  {len(ds):,} ornek -> {yol}")
    return ds


def zaten_var_mi(isim, cikti_klasoru):
    yol = os.path.join(cikti_klasoru, f"{isim}_ds")
    if os.path.exists(yol):
        print(f"  {isim} diskte var, yukleniyor...")
        ds = load_from_disk(yol)
        print(f"    {len(ds):,} ornek\n")
        return ds
    return None


def stream_cek(stream, limit, aciklama, text_fn=None):
    ornekler = []
    pbar = tqdm(total=limit, desc=aciklama)
    for ornek in stream:
        metin = text_fn(ornek) if text_fn else ornek.get("text", "")
        if metin and len(metin.strip()) > 50:
            ornekler.append({"text": metin.strip()})
            pbar.update(1)
        if len(ornekler) >= limit:
            break
    pbar.close()
    return ornekler


def main():
    args = parse_args()
    cikti_klasoru = args.output_dir
    combined_yol = os.path.join(cikti_klasoru, "combined_tr")
    os.makedirs(cikti_klasoru, exist_ok=True)

    if os.path.exists(combined_yol):
        print(f"combined_tr zaten hazir -> {combined_yol}")
        print("Siradaki: python data/clean_data.py")
        return

    print("Turkce veri hazirlaniyor...\n")
    datasetler = []

    print("[1/3] Wikipedia TR...")
    try:
        wiki = load_dataset(
            "wikimedia/wikipedia",
            "20231101.tr",
            split="train",
            streaming=False,
        )
        wiki = wiki.select_columns(["text"])
        datasetler.append(wiki)
        print(f"  {len(wiki):,} makale bulundu\n")
    except Exception as e:
        print(f"  Wikipedia yuklenemedi: {e}\n")

    print("[2/3] CulturaX TR...")
    ds = zaten_var_mi("culturax", cikti_klasoru)
    if ds:
        datasetler.append(ds)
    else:
        try:
            stream = load_dataset("uonlp/CulturaX", "tr", split="train", streaming=True)
            ornekler = stream_cek(stream, args.culturax_limit, "CulturaX TR")
            datasetler.append(kaydet_ve_don("culturax", ornekler, cikti_klasoru))
            print()
        except Exception as e:
            print(f"  CulturaX alinamadi: {e}")
            print("  Gerekirse dataset sayfasindan access request yap.\n")

    print("[3/3] OPUS-100 TR...")
    ds = zaten_var_mi("opus", cikti_klasoru)
    if ds:
        datasetler.append(ds)
    else:
        try:
            stream = load_dataset("Helsinki-NLP/opus-100", "en-tr", split="train", streaming=True)
            ornekler = stream_cek(
                stream,
                args.opus_limit,
                "OPUS TR",
                text_fn=lambda x: x.get("translation", {}).get("tr", ""),
            )
            datasetler.append(kaydet_ve_don("opus", ornekler, cikti_klasoru))
            print()
        except Exception as e:
            print(f"  OPUS alinamadi: {e}\n")

    if not datasetler:
        raise RuntimeError("Hic veri toplanamadi.")

    print(f"\n{len(datasetler)} kaynak birlestiriliyor...")
    combined = concatenate_datasets(datasetler).shuffle(seed=42)
    combined.save_to_disk(combined_yol)

    print("\nTamamlandi")
    print(f"Toplam  : {len(combined):,} ornek")
    print(f"Kayit   : {combined_yol}")
    print("\nSiradaki: python data/clean_data.py")


if __name__ == "__main__":
    main()
