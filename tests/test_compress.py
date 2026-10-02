import numpy as np
import torch
from torch import nn

from compress import TOK, model_bytes, quantize_model, quantize_tensor
from compress.model import GPTConfig, TinyGPT


def small():
    torch.manual_seed(0)
    return TinyGPT(GPTConfig(TOK.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=64))


def test_int8_roundtrip_error():
    w = torch.randn(64, 128)
    q, s = quantize_tensor(w, 8)
    assert (q.float() * s - w).abs().max() <= s.max() / 2 + 1e-6
    assert q.min() >= -128 and q.max() <= 127


def test_int4_group_quantization_range():
    q, s = quantize_tensor(torch.randn(16, 64), 4, group_size=32)
    assert q.min() >= -8 and q.max() <= 7 and s.shape == (16, 2, 1)


def test_quantized_model_close_to_fp32_and_smaller():
    m = small()
    x = torch.randint(0, TOK.vocab_size, (2, 32))
    ref = m(x)[0]
    q8 = quantize_model(m, 8)
    assert (q8(x)[0] - ref).abs().max() < 0.05
    assert model_bytes(q8) < model_bytes(m) and model_bytes(quantize_model(m, 4, 32)) < model_bytes(q8)


def test_onnx_matches_pytorch(tmp_path):
    from compress.onnx_export import export, ort_logits, ort_session

    m = small().eval()
    x = torch.randint(0, TOK.vocab_size, (2, 32))
    sess = ort_session(export(m, str(tmp_path / "m.onnx"), seq_len=32))
    assert np.allclose(ort_logits(sess, x.numpy()), m(x)[0].detach().numpy(), atol=1e-4)
