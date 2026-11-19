"""CLI entry point for Eddi."""

import argparse
from .agent import EddiAgent


def main() -> None:
    parser = argparse.ArgumentParser(prog="eddi", description="Eddi Writable HDC + 1M-Token Context Agent")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    args = parser.parse_args()

    agent = EddiAgent()
    try:
        if args.demo:
            agent.teach("Eddi", "uses", "Hyperdimensional_Computing")
            agent.teach("Eddi", "context_capacity", "1M_tokens_disk_mmap")
            print("Eddi Demo — Writable Knowledge Query:")
            print("  Eddi uses ->", agent.ask("Eddi", "uses"))
            print("  Eddi context_capacity ->", agent.ask("Eddi", "context_capacity"))
        else:
            print("Eddi Agent ready. Run with --demo or import eddi.agent.EddiAgent.")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
