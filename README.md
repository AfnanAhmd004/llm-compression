# llm-compression

Making a language model **smaller and cheaper to serve**: post-training quantization (int8 per-channel, int4 group-wise), **knowledge distillation** into a smaller student, PyTorch dynamic int8 kernels, and **ONNX export with ONNX Runtime**, benchmarked on size, perplexity and CPU latency.

## Techniques

| | Implementation |
|---|---|
| Weight quantization | symmetric int8 per output channel; int4 with one scale per group of 32 input columns; fp16 scales; output head kept in fp32 |
| Dynamic int8 | `torch.ao.quantization.quantize_dynamic`: int8 weights plus int8 matmul kernels with activations quantized at runtime |
| Distillation | student minimises `α·T²·KL(teacher_T ‖ student_T) + (1−α)·CE`, with temperature-softened teacher logits |
| ONNX Runtime | export with dynamic batch, full graph optimisation, numerical parity check against PyTorch |

## Run

```bash
pip install -e ".[dev]"
python examples/benchmark.py     # ~4 min on CPU
pytest
```

Teacher: 4-layer GPT (d=128) trained on a small synthetic language. Student: 2 layers, d=64.

```
model                             size KB  perplexity  latency ms
teacher fp32 (4L, d128)              3155       1.284        8.51
teacher int8 per-channel              860       1.284        9.53
teacher int4 group-32                 515       1.285       10.02
teacher dynamic int8 (kernels)        824       1.284        8.43
student from scratch (2L, d64)        419       1.292        2.90
student distilled (2L, d64)           419       1.287        2.73
teacher via ONNX Runtime             3212      (same)        8.50

ONNX Runtime max |logit diff| vs PyTorch: 2.53e-05
```

### Reading the results

- **Quantization shrinks memory, not necessarily latency.** int4 cuts weights about 6× with negligible perplexity change, but this weight-only version dequantizes in Python on every call, so it is *slower*. Real speed-ups need fused low-bit kernels (as in GPTQ/AWQ runtimes, TensorRT-LLM or llama.cpp). At this tiny size, even PyTorch's int8 kernels barely move latency, because overheads dominate.
- **Distillation buys speed.** The student is 7.5× smaller and 3× faster. Training it on the teacher's soft targets gives lower perplexity than training the same architecture from scratch.
- **ONNX export is exact** to about 1e-5, which makes it a safe starting point for TensorRT or edge runtimes.

The synthetic language is easy, so all perplexities are close to 1. Apply the same harness to a real corpus to separate the methods more clearly.

## License

MIT
