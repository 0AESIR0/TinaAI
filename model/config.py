from dataclasses import dataclass, field
import json

@dataclass
class ModelConfig:
    # Vocab
    vocab_size:               int   = 32000
    pad_token_id:             int   = 0
    bos_token_id:             int   = 1
    eos_token_id:             int   = 2

    # 125M parametreye göre ayarlandı — 4GB VRAM'e sığar
    hidden_size:              int   = 768
    num_hidden_layers:        int   = 12
    num_attention_heads:      int   = 12
    intermediate_size:        int   = 3072
    max_position_embeddings:  int   = 1024
    hidden_dropout_prob:      float = 0.1
    attention_dropout_prob:   float = 0.1

    model_type:               str   = "Tina"
    architectures:            list  = field(default_factory=lambda: ["Tina"])

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        alanlari = {f.name for f in __import__("dataclasses").fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in alanlari})

    def kaydet(self, yol: str):
        with open(yol, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def yukle(cls, yol: str):
        with open(yol, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))