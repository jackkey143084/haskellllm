# tiny-llm in Haskell — plan

Goal: port tiny-qwen (Qwen3.5-0.8B text path) to pure Haskell, token-for-token validated
against the PyTorch reference we already have. Reuse every hard-won lesson from the Go port.
Repo: new local dir /workspace/haskell-qwen (push to GitHub via Git Data API when Esi says go).

## Phase 0 — toolchain (sandbox)
- `apt install ghc cabal-install` → GHC 9.6.6, confirmed available in trixie.
- Compile-time gotcha: 3GB cgroup. Build with `-j1` first; GHC itself can OOM on big modules.
- Dep policy (recommend): cabal + minimal deps — `bytestring`, `vector`, `aeson` (safetensors
  header), `unix` (mmap). Everything else hand-rolled, matching the pure-Go ethos.
  Zero-dep variant possible (hand-parse the safetensors JSON header) if Esi wants purity.

## Phase 1 — substrate
- **bf16 type**: store as `Word16`, convert to f32 by `w << 16` reinterpret (decode) and
  round-to-nearest-even f32→bf16 (encode). GHC `half` package is IEEE half, NOT bf16 — don't use it.
- **safetensors reader**: little-endian JSON header + aligned raw tensor data.
  mmap-backed reads (mmap is load-bearing on this box — heap copies OOM at 2x peak).
- **Config**: parse the JSON model config (n_layers=..., head_dim, kv heads, etc).
- Unit tests: bf16 round-trip vs Python struct output, header offsets.

## Phase 2 — tokenizer
- Port BPE + special-token handling; parity vs PyTorch tokenizer on a fixed prompt set.
- Preserve the enable_thinking prompt-append behavior: TRUE → only `" thinking\n"`,
  FALSE → the longer block. Match the mode before chasing a divergence (learned the hard way).

## Phase 3 — forward pass + parity harness (the real work)
- RMSNorm → RoPE → GQA attention (with KV cache) → SwiGLU MLP, bf16 weights / f32 accum.
- Parity harness FIRST, not after: per-layer f32 activation dumps from both sides
  (reuse the PyTorch dump scripts from /tmp if they survive, else regenerate — torch is
  reinstallable via system pip but EPHEMERAL).
- Debug discipline (from vision + text parity debugging):
  - Confirm with element-wise max-abs-diff on IDENTICAL dumped input before trusting checksums;
    checksums lie when norm weights are huge.
  - Compare PyTorch's own bf16-vs-f32 drift first to set the noise floor.
  - geluTanh constant is sqrt(2/pi) = 0.7979, NOT sqrt(2)/pi. Write the test on day one.
- Known-expected wrinkle: FP near-tie flips at low-logit-gap steps (Go had one at step 6/5).
  Decide policy up front: token-for-token except near-ties, or exact.

## Phase 4 — decode loop + validation
- Greedy decode, KV cache, streaming. Bar: 16/16 token match on the standard fixed prompt
  (same as the Go bar). Then a second prompt for insurance.

## Phase 5 — CLI + (optional) server
- Chat REPL. Careful: the phase-5 qwen-go REPL executes shell — build read-only here,
  shell-exec only in interactive sessions with Esi driving.
- Optional: OpenAI-compatible /v1/chat/completions server (warp), mirroring qwen-openai.

## Phase 6 — perf
- Lessons from Go apply: scalar GEMV is at its ceiling fast; unroll/blocking variants LOSE to
  the simple loop. First wins: strictness (avoid thunk buildup — Haskell's version of Go's GC
  churn), unboxed vectors, then thread-parallel output rows. Don't chase SIMD.
- int8 quantization = size/memory win, not speed (measured in Go). Only if wanted.

## Deferred / out of scope
- Vision tower: only after text is validated. PNG-only parity (JPEG decoders can never match),
  and the PIL bicubic verbatim-Resample.c recipe if we get there.
- mRoPE: text path first, image positions later.

## Order
0 → 1 → 2 → 3 → 4 straight through (harness built during 3). 5/6 after Esi sees a working decode.
