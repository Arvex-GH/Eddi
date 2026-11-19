"""Command line interface for Eddi.

    python -m eddi.cli              # status
    python -m eddi.cli --demo       # teach a few facts and query them
    python -m eddi.cli --train      # train the built-in curriculum, save brain
    python -m eddi.cli --train --corpus ./books --epochs 5
    python -m eddi.cli --bench      # held-out benchmark of the saved brain
    python -m eddi.cli --chat       # interactive chat (memory + reasoning + LM)
    python -m eddi.cli --ask "what is the moon of earth"
    python -m eddi.cli --serve      # tiny JSON HTTP API
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__, curriculum, evaluation
from .agent import EddiAgent
from .paths import default_brain_path, find_core_binary
from .training import (
    TrainingConfig,
    brain_exists,
    load_brain,
    save_brain,
    train,
)


def _load_agent(brain: Optional[str], quiet: bool = False) -> EddiAgent:
    if brain_exists(brain):
        agent = load_brain(brain)
        if not quiet:
            print(f"Loaded brain: {brain or default_brain_path()}")
        return agent
    if not quiet:
        print("No trained brain found - using a fresh agent (run --train first).")
    return EddiAgent()


def cmd_status(args: argparse.Namespace) -> int:
    path = args.brain or default_brain_path()
    print(f"Eddi v{__version__}")
    print(f"Brain file : {path} ({'found' if os.path.exists(path) else 'missing'})")
    core = find_core_binary()
    print(f"C++ core   : {core if core else 'not built (optional, python fallback is used)'}")
    if os.path.exists(path):
        agent = load_brain(path)
        try:
            stats = agent.stats()
            print("Knowledge  :")
            for key, value in stats.items():
                print(f"  {key:<16}: {value}")
            if agent.lm.trained_tokens:
                print(
                    "  held-out ppl   : "
                    f"{agent.lm.perplexity(evaluation.HELD_OUT_SENTENCES, skip_oov=True):.2f}"
                )
        finally:
            agent.close()
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    agent = EddiAgent(persist_context=False)
    try:
        agent.teach_many(
            [
                ("Eddi", "uses", "Hyperdimensional_Computing"),
                ("Eddi", "context_capacity", "1M_tokens_disk_mmap"),
                ("Earth", "orbits", "Sun"),
                ("Moon", "orbits", "Earth"),
            ]
        )
        print("Eddi demo - writable memory (no training required):")
        for subject, relation in (("Eddi", "uses"), ("Eddi", "context_capacity"), ("Moon", "orbits")):
            print(f"  {subject} --{relation}-->", agent.ask(subject, relation))
        print("\nReasoning demo:")
        agent.ingest_text(
            "Eddi is a local hyperdimensional agent that runs on your own computer. "
            "It writes facts into memory without backpropagation."
        )
        for question in (
            "Earth is to Sun as Moon is to ?",
            "chain Moon | orbits | orbits",
            "what is 17 * 3 + 4",
            "Eddi is a local",
        ):
            answer = agent.answer(question)
            print(f"  [{answer.kind}] {question}\n      -> {answer.text}")
    finally:
        agent.close()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config = TrainingConfig(
        packs=tuple(args.pack) if args.pack else curriculum.ALL_PACKS,
        corpus_dirs=tuple(args.corpus or ()),
        corpus_glob=args.corpus_glob,
        epochs=args.epochs,
        order=args.order,
        evaluate=not args.no_eval,
        prune_min_count=args.prune,
        save_path=args.brain or default_brain_path(),
        max_corpus_tokens=args.max_tokens,
    )
    agent = EddiAgent(lm_order=args.order, persist_context=False)
    try:
        report = train(agent, config, verbose=True)
        print()
        print(report.pretty())
        print(f"\nSaved to {report.saved_to}")
    finally:
        agent.close()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    agent = _load_agent(args.brain)
    try:
        metrics = evaluation.run_benchmark(agent)
        print(evaluation.format_benchmark(metrics))
        _, fact_misses = evaluation.evaluate_facts(agent)
        _, reasoning_misses = evaluation.evaluate_reasoning(agent)
        if fact_misses or reasoning_misses:
            print("\nMisses:")
            for miss in fact_misses + reasoning_misses:
                print(f"  - {miss}")
        if args.json:
            print(json.dumps(metrics, indent=2))
    finally:
        agent.close()
    return 0 if metrics.get("fact_accuracy", 0) >= 0.8 else 1


def cmd_ask(args: argparse.Namespace) -> int:
    agent = _load_agent(args.brain, quiet=True)
    try:
        answer = agent.answer(args.ask)
        print(answer.text)
        if args.explain and answer.details:
            for detail in answer.details:
                print(f"  · {detail}")
    finally:
        agent.close()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    agent = _load_agent(args.brain)
    print("Eddi chat. 'help' for commands, 'quit' to exit.")
    try:
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.lower() in {"quit", "exit", "q"}:
                break
            if not line:
                continue
            answer = agent.answer(line)
            print(f"eddi [{answer.kind}]> {answer.text}")
    finally:
        if args.save_on_exit:
            save_brain(agent, args.brain)
            print(f"Brain saved to {args.brain or default_brain_path()}")
        agent.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import http.server
    import socketserver

    agent = _load_agent(args.brain, quiet=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/health"):
                self._send({"status": "ok", "version": __version__})
            elif self.path.startswith("/stats"):
                self._send(agent.stats())
            else:
                self._send({"error": "use POST /ask with {\"text\": \"...\"}"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/ask"):
                self._send({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send({"error": "invalid json"}, 400)
                return
            text = str(payload.get("text", "")).strip()
            if not text:
                self._send({"error": "missing text"}, 400)
                return
            answer = agent.answer(text)
            self._send(
                {"text": answer.text, "kind": answer.kind, "confidence": answer.confidence}
            )

        def log_message(self, fmt: str, *args) -> None:  # silence default logging
            pass

    try:
        with socketserver.ThreadingTCPServer((args.host, args.port), Handler) as httpd:
            print(f"Eddi API listening on http://{args.host}:{args.port} (POST /ask)")
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        agent.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eddi",
        description=f"Eddi v{__version__} - writable HDC memory + trained language model + 1M-token context",
    )
    parser.add_argument("--version", action="version", version=f"eddi {__version__}")
    parser.add_argument("--brain", default=None, help="path to the saved brain .pkl")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="run the capability demo")
    mode.add_argument("--train", action="store_true", help="train the curriculum/corpus and save the brain")
    mode.add_argument("--bench", action="store_true", help="run the held-out benchmark")
    mode.add_argument("--chat", action="store_true", help="interactive chat")
    mode.add_argument("--serve", action="store_true", help="serve a JSON HTTP API")
    mode.add_argument("--ask", metavar="QUESTION", help="answer one question and exit")

    parser.add_argument("--pack", action="append", help="curriculum pack (repeatable)")
    parser.add_argument("--corpus", action="append", help="directory of text files to learn (repeatable)")
    parser.add_argument("--corpus-glob", default="*.txt", help="file pattern for --corpus (default *.txt)")
    parser.add_argument("--epochs", type=int, default=3, help="training passes over the corpus (default 3)")
    parser.add_argument("--order", type=int, default=3, help="context length + 1 (default 3)")
    parser.add_argument("--max-tokens", type=int, default=2_000_000, help="corpus token budget")
    parser.add_argument("--prune", type=int, default=1, help="drop n-grams seen fewer than N times")
    parser.add_argument("--no-eval", action="store_true", help="skip the held-out benchmark during training")
    parser.add_argument("--json", action="store_true", help="print benchmark as JSON")
    parser.add_argument("--explain", action="store_true", help="show where the answer came from")
    parser.add_argument("--save-on-exit", action="store_true", help="save the brain when chat exits")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.train:
        return cmd_train(args)
    if args.bench:
        return cmd_bench(args)
    if args.chat:
        return cmd_chat(args)
    if args.serve:
        return cmd_serve(args)
    if args.ask:
        return cmd_ask(args)
    if args.demo:
        return cmd_demo(args)
    return cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())
