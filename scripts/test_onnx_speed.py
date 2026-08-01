"""
One-off isolated test: does the ONNX (Xenova/bge-m3) path actually beat the
current sentence-transformers/PyTorch CPU baseline for embedding a single
query? Not part of the pipeline - just answering the speed question.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

QUERY = "why did the Quraysh want to fight at Uhud"
MODEL_ID = "Xenova/bge-m3"

print(f"Loading {MODEL_ID} via ONNX Runtime (CPUExecutionProvider)...")
t0 = time.perf_counter()
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, provider="CPUExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
load_time = time.perf_counter() - t0
print(f"Load time: {load_time:.2f}s")

def embed(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    cls = outputs.last_hidden_state[:, 0, :].detach().numpy()[0]
    normed = cls / np.linalg.norm(cls)
    return normed

# warm-up call (first inference often includes extra graph-optimization overhead)
_ = embed(QUERY)

# timed runs
times = []
for _ in range(5):
    t0 = time.perf_counter()
    vec = embed(QUERY)
    times.append(time.perf_counter() - t0)

print(f"\nEmbedding dim: {vec.shape}")
print(f"Per-query encode times over 5 runs (ms): {[round(t*1000, 1) for t in times]}")
print(f"Average (excluding warm-up): {sum(times)/len(times)*1000:.1f} ms")
print(f"\nFor comparison, sentence-transformers/PyTorch baseline was: 3988.9 ms")
