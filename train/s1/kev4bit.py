"""Kev (System One w stylu Jev) na karcie 8 GB: baza Qwen3.5 w NF4 (bitsandbytes) + adapter LoRA Kev + pointer head.

Kod Kev (data/external/kev_src/kev, Apache-2.0) wczytuje baze w fp32/bf16, co dla 4B oznacza 8-16 GB.
Tu baza jest kwantyzowana do 4 bitow (QLoRA), a cala reszta - kodowanie zadania (kev.model.encode),
wiersze pytan dla warstw Gated DeltaNet (rows_of / _rows_hidden), pointer head i temperatura - pochodzi z Kev,
wiec odpowiedzi sa liczone dokladnie tym samym algorytmem.

    from train.s1.kev4bit import load_kev
    tok, model = load_kev("jaredpalmer/kev-4b")          # eval, 4-bit
    tok, model = load_kev("jaredpalmer/kev-4b", train=True)  # adapter + head trenowalne (QLoRA)
"""
from __future__ import annotations

import gc

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from kev.checkpoint import Checkpoint
from kev.model import DecisionModel, PointerHead, is_hybrid, load_tokenizer, pad_id


class QuantDecisionModel(DecisionModel):
    """DecisionModel z backbonem w NF4. Dziedziczy encode/forward/probs/prefix z Kev bez zmian."""

    def __init__(self, lm, tok, head_dim, device):
        torch.nn.Module.__init__(self)
        self.lm = lm
        self.pad_id = pad_id(tok)
        self.hybrid = is_hybrid(lm.config)
        self.option_isolation = False
        self.head = PointerHead(lm.config.hidden_size, dp=head_dim).to(device)
        self.device = device


def load_kev(run="jaredpalmer/kev-4b", device="cuda", train=False, compute_dtype=torch.bfloat16, gradient_checkpointing=True):
    """-> (tokenizer, QuantDecisionModel). train=False: eval mode, temperatura z head.pt. train=True: LoRA i head
    trenowalne, backbone zamrozony w NF4 (QLoRA), gradient checkpointing."""
    from peft import PeftModel, prepare_model_for_kbit_training

    ck = Checkpoint(run)
    meta = ck.meta
    if ck.full:
        raise ValueError("checkpoint z pelnymi wagami - ta sciezka obsluguje tylko adaptery LoRA")
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=compute_dtype)
    causal = AutoModelForCausalLM.from_pretrained(meta.base, revision=meta.base_revision, quantization_config=bnb,
                                                  dtype=compute_dtype, device_map={"": device},
                                                  attn_implementation="sdpa")
    lm = causal.model                  # sam backbone, bez glowicy slownika (Kev nigdy nie generuje tekstu)
    del causal.lm_head
    del causal
    gc.collect(); torch.cuda.empty_cache()
    if train:
        lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=gradient_checkpointing)
    lm = PeftModel.from_pretrained(lm, str(ck.path), is_trainable=train)
    model = QuantDecisionModel(lm, tok, meta.head_dim, device)
    model.head.load_state_dict(meta.head)
    model.head.temperature = meta.temperature
    model.train(train)
    if not train:
        model.eval()
    return tok, model


def memory_report(tag=""):
    a = torch.cuda.memory_allocated() / 2**30
    p = torch.cuda.max_memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    return f"[{tag}] VRAM: zajete {a:.2f} GB, szczyt {p:.2f} GB, zarezerwowane {r:.2f} GB"
