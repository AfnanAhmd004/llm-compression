"""Teacher vs quantized vs distilled vs ONNX Runtime: size, perplexity and CPU latency."""
import os
import tempfile
import time

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from compress import TOK, corpus, distill, model_bytes, perplexity, quantize_model, train_lm
from compress.model import GPTConfig, TinyGPT
from compress.onnx_export import export, ort_logits, ort_session

torch.manual_seed(0)
train_text, test_text = corpus(4000, 1), corpus(400, 2)
teacher = train_lm(TinyGPT(GPTConfig(TOK.vocab_size, block_size=64, n_layer=4, n_head=4, n_embd=128)), train_text, steps=1500)
student = distill(teacher, TinyGPT(GPTConfig(TOK.vocab_size, block_size=64, n_layer=2, n_head=4, n_embd=64)), train_text, steps=1500)
scratch = train_lm(TinyGPT(GPTConfig(TOK.vocab_size, block_size=64, n_layer=2, n_head=4, n_embd=64)), train_text, steps=1500)

x = torch.randint(0, TOK.vocab_size, (8, 64))


def latency_ms(fn, reps=30):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


rows = [
    ("teacher fp32 (4L, d128)", teacher),
    ("teacher int8 per-channel", quantize_model(teacher, 8)),
    ("teacher int4 group-32", quantize_model(teacher, 4, 32)),
    ("teacher dynamic int8 (kernels)", torch.ao.quantization.quantize_dynamic(teacher, {torch.nn.Linear}, dtype=torch.qint8)),
    ("student from scratch (2L, d64)", scratch),
    ("student distilled (2L, d64)", student),
]
print(f"{'model':<32}{'size KB':>9}{'perplexity':>12}{'latency ms':>12}")
for name, m in rows:
    m.eval()
    with torch.no_grad():
        lat = latency_ms(lambda: m(x))
    print(f"{name:<32}{model_bytes(m) / 1024:>9.0f}{perplexity(m, test_text):>12.3f}{lat:>12.2f}")

with tempfile.TemporaryDirectory() as d:
    path = export(teacher, os.path.join(d, "teacher.onnx"))
    sess = ort_session(path)
    diff = np.abs(ort_logits(sess, x.numpy()) - teacher(x)[0].detach().numpy()).max()
    print(f"{'teacher via ONNX Runtime':<32}{os.path.getsize(path) / 1024:>9.0f}{'(same)':>12}{latency_ms(lambda: ort_logits(sess, x.numpy())):>12.2f}")
    print(f"\nONNX Runtime max |logit diff| vs PyTorch: {diff:.2e}")
