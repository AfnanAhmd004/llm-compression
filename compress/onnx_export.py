"""Export to ONNX and run with ONNX Runtime."""
from __future__ import annotations

import numpy as np
import torch


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx):
        return self.model(idx)[0]


def export(model, path: str, seq_len: int = 64) -> str:
    model.eval()
    dummy = torch.zeros(1, seq_len, dtype=torch.long)
    torch.onnx.export(_LogitsOnly(model), (dummy,), path, input_names=["idx"], output_names=["logits"],
                      dynamic_axes={"idx": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    return path


def ort_session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def ort_logits(sess, idx: np.ndarray) -> np.ndarray:
    return sess.run(["logits"], {"idx": idx.astype(np.int64)})[0]
