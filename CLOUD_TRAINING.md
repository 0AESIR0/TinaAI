# Kaggle ve Colab Egitim Rehberi

Bu proje artik Kaggle ve Colab ortamlarinda komut satirindan daha rahat calisacak sekilde parametreli hale getirildi.

## 1. Kaggle icin hizli akıs

Notebook ayarlari:
- `Accelerator`: GPU
- `Internet`: On
- `Persist output`: On

Hucre 1:

```bash
!git clone <SENIN_REPO_URL> TinaAI
%cd TinaAI
!pip install -r requirements.txt
```

Hucre 2:

```bash
!python data/download_data.py --output-dir /kaggle/working/data
!python data/clean_data.py --input-dir /kaggle/working/data/combined_tr --output-dir /kaggle/working/data/cleaned_tr
!python tokenizer/train_tokenizer.py --data-dir /kaggle/working/data/cleaned_tr --output-dir /kaggle/working/tokenizer/tr_tokenizer --sample-count 300000 --vocab-size 32000
```

Hucre 3:

```bash
!python training/train.py \
  --data-dir /kaggle/working/data/cleaned_tr \
  --tokenizer-dir /kaggle/working/tokenizer/tr_tokenizer \
  --output-dir /kaggle/working/checkpoints \
  --batch-size 8 \
  --grad-accum 8 \
  --max-len 1024 \
  --epochs 3 \
  --save-every 500 \
  --max-samples 400000
```

Kaggle oturumu kesilirse devam:

```bash
!python training/train.py \
  --data-dir /kaggle/working/data/cleaned_tr \
  --tokenizer-dir /kaggle/working/tokenizer/tr_tokenizer \
  --output-dir /kaggle/working/checkpoints \
  --resume-from /kaggle/working/checkpoints/final \
  --batch-size 8 \
  --grad-accum 8 \
  --max-len 1024 \
  --epochs 5
```

Not:
- Kaggle ciktilari `Output` olarak saklanabildigi icin checkpoint klasorunu indirip sonra yeniden kullanabilirsin.
- `best` klasoru genelde inference icin en guvenli secimdir.

## 2. Colab icin hizli akis

Hucre 1:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Hucre 2:

```bash
%cd /content
!git clone <SENIN_REPO_URL> TinaAI
%cd TinaAI
!pip install -r requirements.txt
```

Hucre 3:

```bash
!python data/download_data.py --output-dir /content/drive/MyDrive/TinaAI/data
!python data/clean_data.py --input-dir /content/drive/MyDrive/TinaAI/data/combined_tr --output-dir /content/drive/MyDrive/TinaAI/data/cleaned_tr
!python tokenizer/train_tokenizer.py --data-dir /content/drive/MyDrive/TinaAI/data/cleaned_tr --output-dir /content/drive/MyDrive/TinaAI/tokenizer/tr_tokenizer
```

Hucre 4:

```bash
!python training/train.py \
  --data-dir /content/drive/MyDrive/TinaAI/data/cleaned_tr \
  --tokenizer-dir /content/drive/MyDrive/TinaAI/tokenizer/tr_tokenizer \
  --output-dir /content/drive/MyDrive/TinaAI/checkpoints \
  --batch-size 6 \
  --grad-accum 8 \
  --max-len 1024 \
  --epochs 3 \
  --save-every 500
```

Devam etmek icin:

```bash
!python training/train.py \
  --data-dir /content/drive/MyDrive/TinaAI/data/cleaned_tr \
  --tokenizer-dir /content/drive/MyDrive/TinaAI/tokenizer/tr_tokenizer \
  --output-dir /content/drive/MyDrive/TinaAI/checkpoints \
  --resume-from /content/drive/MyDrive/TinaAI/checkpoints/final \
  --batch-size 6 \
  --grad-accum 8 \
  --max-len 1024 \
  --epochs 5
```

## 3. GPU'ya gore pratik ayar

- T4 / 15 GB: `--batch-size 6` veya `8`, `--max-len 1024`
- L4 / A10: `--batch-size 8` veya `12`, `--max-len 1024`
- Kucuk GPU: `--batch-size 2`, `--grad-accum 16`, gerekirse `--max-len 512`

## 4. Daha guclu model icin

- Ilk asamada ayni mimariyle daha cok veri ve daha uzun sure egitmek genelde en iyi kazanimi verir.
- Sonra istersen `--hidden-size`, `--num-layers`, `--num-heads`, `--intermediate-size` ile modeli buyutebilirsin.
- Modeli buyutursan VRAM ihtiyaci hizla artar; once veri miktarini ve epoch sayisini artirmak daha verimli olur.
