# Architecture decisions

Short records of the choices that shape this project. Each lists the context, the decision and what it costs.

## ADR-1: One SQLite file for vectors, BM25 and citable text

**Context.** The target buyer is a small or mid-sized team that wants "chat with our documents" without running a vector database.

**Decision.** Store documents, citable sections, chunks, float32 embeddings and an FTS5 BM25 index in one SQLite file. Dense search loads the matrix into NumPy and does a dot product.

**Consequences.** No infrastructure, the index can be copied around or backed up as one file, and BM25 comes from SQLite itself. A brute-force scan is fine up to roughly 10^5 chunks; past that, swap `dense_search` for an ANN index (sqlite-vec, FAISS, pgvector). The retriever only depends on two functions, so the swap stays local.

## ADR-2: Local ONNX models, no PyTorch

**Context.** sentence-transformers normally pulls in PyTorch (hundreds of MB to GBs).

**Decision.** Run the official ONNX exports of `sentence-transformers/all-MiniLM-L6-v2` (embeddings) and `cross-encoder/ms-marco-MiniLM-L6-v2` (reranking) with `onnxruntime` and `tokenizers`.

**Consequences.** Small install, CPU only, no GPU needed. Both models are English-only and small; a multilingual corpus or harder questions would want a bigger embedder (e.g. bge-m3) and reranker. Models are fetched once from the Hugging Face Hub.

## ADR-3: JSON citations plus code verification, not free-text references

**Context.** A model can cite a source that does not say what it claims. Users of a document assistant need to trust each sentence.

**Decision.** The answer comes back through structured outputs (`output_config.format` JSON schema) as `{answerable, answer, citations[{source, quote}]}`. Every quote is then checked in code: it must occur in the passage shown to the model and in the stored text of the cited PDF page or section, after a light normalisation (case, whitespace, curly quotes, dash variants). Unverified citations are flagged in the UI and counted in the eval.

**Consequences.** Faithfulness is measured, not assumed. The check is strict about words and numbers, so a model that paraphrases inside a quote is penalised even when the claim is right. Anthropic's built-in Citations feature was not used because it cannot be combined with structured outputs, and the abstention flag is easier to evaluate as a field.

## ADR-4: Abstain as a first-class answer

**Decision.** The schema has an explicit `answerable` flag and the prompt tells the model to set it to false when the passages do not contain the answer, rather than guessing numbers or rules. The eval has unanswerable questions that sit close to the corpus topics (for example a fine amount the holiday guide mentions but never states).

**Consequences.** Abstention accuracy and false-abstention rate are both reported, because a system that always abstains would score perfectly on the first.

## ADR-5: Anthropic SDK, any Anthropic-compatible endpoint

**Context.** Teams reach Claude in different ways: directly, or through a company gateway or router that speaks the Anthropic Messages API.

**Decision.** All model calls go through the official `anthropic` Python SDK behind a small `LLMClient` protocol. The base URL (`RAG_BASE_URL`), the name of the env var holding the key (`RAG_API_KEY_ENV`) and the model ids (`RAG_ANSWER_MODEL`, `RAG_JUDGE_MODEL`) are configuration. Defaults are the direct Anthropic API and ids (`claude-sonnet-5-5`). On the direct API, Sonnet/Opus requests also opt into server-side refusal fallbacks.

**Consequences.** The published eval ran through an Anthropic-compatible gateway (OpenRouter, ids `anthropic/claude-sonnet-5.5` and `anthropic/claude-haiku-5.5`) because the direct API key on the build machine was rejected. Refusal fallbacks are only requested for direct Anthropic model ids. A second backend, `RAG_LLM_BACKEND=claude-cli`, sends the same prompt and schema through a signed-in Claude Code CLI for machines with no API key at all; tests mock both backends.
