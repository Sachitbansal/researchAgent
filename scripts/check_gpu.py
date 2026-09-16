#!/usr/bin/env python3
"""
Reports whether the GPU path is live, and times the three models that would use it.

In:  nothing (optional --skip-bench). Out: printed device, driver state and timings;
exit 0 when CUDA is in use, 1 when the run fell back to CPU. Run it before and after
installing the NVIDIA driver — the numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import CFG  # noqa: E402
from common.device import cuda_available, resolve_device  # noqa: E402


def _driver_report() -> None:
    """What torch can see, and why it cannot see a GPU when it cannot."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        print(f"  torch            NOT IMPORTABLE: {exc}")
        return

    print(f"  torch            {torch.__version__}")
    print(f"  built for CUDA   {torch.version.cuda or 'cpu-only wheel'}")
    print(f"  cuda.is_available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  device {i}         {props.name}, "
                  f"{props.total_memory / 1024**3:.1f} GiB, sm_{props.major}{props.minor}")
    else:
        # The common case on a fresh machine: CUDA wheel, no host driver.
        note = "no NVIDIA kernel driver loaded" if not Path("/proc/driver/nvidia").exists() \
            else "driver present but torch cannot use it (version mismatch?)"
        print(f"  reason           {note}")


def _corpus_texts() -> list:
    """Real chunk text, because synthetic text makes the GPU look better than it is.

    sentence-transformers sorts a batch by length before padding it, so 406 copies of one
    short sentence pack into ~30-token batches while the real corpus averages 278 tokens.
    Benchmarking the former measured about a ninth of the actual work and reported a 63x
    speedup where the true figure on this corpus is 7x. Falls back to synthetic text of a
    representative length only when no corpus has been indexed yet.
    """
    try:
        from corpus.chunk_store import ChunkStore

        texts = [chunk["text"] for chunk in ChunkStore(config=CFG).all()]
        if texts:
            return texts
    except Exception:
        pass
    filler = ("Mixture-of-experts routing sends each token to a small subset of experts, "
              "which keeps the parameter count high while holding inference cost flat. ")
    return [filler * 14] * 406  # ~278 tokens each, the measured corpus mean


def _bench(device: str) -> None:
    """Time the bi-encoder, the reranker and the NLI model on whatever device resolved."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    texts = _corpus_texts()

    t = time.perf_counter()
    bi = SentenceTransformer(str(CFG.embedding.model), device=device)
    print(f"  bi-encoder load   {time.perf_counter() - t:6.2f}s")
    t = time.perf_counter()
    bi.encode(texts, batch_size=int(CFG.embedding.batch_size), convert_to_numpy=True,
              normalize_embeddings=True, show_progress_bar=False)
    print(f"  EMBED {len(texts)} chunks {time.perf_counter() - t:6.2f}s   (index time)")

    ce = CrossEncoder(str(CFG.retrieval.reranker_model), device=device)
    k = int(CFG.retrieval.k_retrieve)
    # Distinct chunks, for the same reason _corpus_texts exists: identical pairs pad into
    # one uniform batch and flatter the GPU.
    pool = (texts * (k // max(1, len(texts)) + 1))[:k]
    pairs = [("how does expert routing affect accuracy", t[:2000]) for t in pool]
    ce.predict(pairs)  # warm up; the first pass pays for kernel compilation
    t = time.perf_counter()
    ce.predict(pairs)
    print(f"  RERANK {len(pairs)} pairs  {time.perf_counter() - t:6.2f}s   (per query)")

    nli = CrossEncoder(str(CFG.nli.model), device=device)
    n = int(CFG.nli.max_pairs)
    sentences = [s for t in texts[:20] for s in t.split(". ")][: n + 1] or [texts[0]] * (n + 1)
    nli_pairs = [(sentences[i][:2000], sentences[i + 1][:2000])
                 for i in range(min(n, len(sentences) - 1))]
    nli.predict(nli_pairs)
    t = time.perf_counter()
    nli.predict(nli_pairs)
    print(f"  NLI {len(nli_pairs)} pairs     {time.perf_counter() - t:6.2f}s   (per check)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-bench", action="store_true",
                        help="report the device only; do not download or time models")
    args = parser.parse_args()

    print("driver / torch")
    _driver_report()

    print("\nconfig")
    print(f"  compute.device   {CFG.compute.device}")
    device = resolve_device(CFG)
    print(f"  resolved to      {device}")

    if not args.skip_bench:
        print(f"\nbenchmark on {device}")
        _bench(device)

    if device.startswith("cuda"):
        print("\nGPU is in use.")
        return 0
    print("\nRunning on CPU. Install the NVIDIA driver to use the GPU; "
          "compute.device is 'auto', so nothing else needs to change.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
