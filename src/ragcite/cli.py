"""Command line: `ragcite ingest | ask | eval | serve`."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ragcite.config import Settings
from ragcite.evaluation import (
    evaluate_question,
    evaluate_retrievers,
    load_questions,
    results_to_json,
    summarize,
)
from ragcite.report import pct, render_html, render_markdown


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    """Build or refresh the index from a folder of documents."""
    from ragcite.ingest import ingest_folder
    from ragcite.pipeline import embedder_for
    from ragcite.store import connect

    stats = ingest_folder(
        connect(settings.db_path), Path(args.folder), embedder_for(settings.embed_model), max_words=args.chunk_words
    )
    print(
        f"indexed {stats.documents} documents, {stats.sections} sections, {stats.chunks} chunks into {settings.db_path}"
    )
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    """Answer one question and print the answer with its verified citations."""
    from ragcite.pipeline import build_answerer, open_index

    answer = build_answerer(settings, open_index(settings)).answer(args.question)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(answer.text + "\n")
    for c in answer.citations:
        status = "verified" if c.verified else "NOT FOUND ON PAGE"
        print(f'[{c.number}] {c.title} - {c.location} ({status})\n    "{c.quote}"')
    return 0


def retrieval_variants(settings: Settings, conn) -> dict:
    """The retrieval ablation: each first-stage mode, with and without reranking."""
    from ragcite.pipeline import build_retriever

    base = build_retriever(settings, conn).with_config(final_k=5)
    return {
        "BM25 only": base.with_config(mode="bm25", rerank_k=0),
        "Dense only (MiniLM)": base.with_config(mode="dense", rerank_k=0),
        "Hybrid (BM25 + dense, RRF)": base.with_config(rerank_k=0),
        "Hybrid + cross-encoder rerank (default)": base,
    }


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    """Run the retrieval ablation and the full answer eval, then write the reports."""
    from ragcite.llm import make_llm_client
    from ragcite.pipeline import build_answerer, open_index

    conn = open_index(settings)
    items = load_questions(Path(args.questions))[: args.limit or None]
    retrieval = evaluate_retrievers(items, retrieval_variants(settings, conn))
    llm = make_llm_client(**settings.llm_kwargs())
    answerer = build_answerer(settings, conn, llm)
    results = []
    for index, item in enumerate(items, start=1):
        result = evaluate_question(item, answerer.answer, llm, settings.judge_model)
        results.append(result)
        outcome = "abstained" if result.answer["abstained"] else "answered"
        print(f"[{index}/{len(items)}] {item.id}: {outcome} {result.verdict}".rstrip())
    write_reports(Path(args.out), args.name, settings, results, retrieval)
    return 0


def write_reports(out: Path, name: str, settings: Settings, results: list, retrieval: dict) -> None:
    """Write results.json, report.md and report.html into out/ and print the headline numbers."""
    meta = {
        "run_name": name,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "answer_model": settings.answer_model,
        "judge_model": settings.judge_model,
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "final_k": settings.final_k,
        "llm_backend": settings.llm_backend,
    }
    summary = summarize(results)
    run = results_to_json(summary, results, meta) | {"retrieval": retrieval}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    (out / "report.md").write_text(render_markdown(run, retrieval))
    (out / "report.html").write_text(render_html(run, retrieval))
    print(
        f"correctness {pct(summary.correctness)} | faithfulness {pct(summary.citation_faithfulness)} | "
        f"abstention {pct(summary.abstention_accuracy)} | wrote {out}/report.md and report.html"
    )


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    """Start the web UI."""
    import uvicorn

    from ragcite.web.app import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI."""
    parser = argparse.ArgumentParser(prog="ragcite", description="Chat with your documents, with verified citations.")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="index a folder of PDF/Markdown/HTML files")
    ingest.add_argument("folder")
    ingest.add_argument("--chunk-words", type=int, default=180)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true")
    run = sub.add_parser("eval", help="run the eval suite and write Markdown + HTML reports")
    run.add_argument("--questions", default="evals/questions.jsonl")
    run.add_argument("--out", default="docs/eval")
    run.add_argument("--name", default="default")
    run.add_argument("--limit", type=int, default=0)
    serve = sub.add_parser("serve", help="start the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


COMMANDS = {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval, "serve": cmd_serve}


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `ragcite` console script."""
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args, Settings())


if __name__ == "__main__":
    sys.exit(main())
