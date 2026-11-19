"""Dataset Ingestion & Knowledge Teaching Pipeline for Eddi."""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from .agent import EddiAgent


CORE_FACTS: List[Tuple[str, str, str]] = [
    ("Earth", "orbits", "Sun"),
    ("Moon", "orbits", "Earth"),
    ("Water", "chemical_formula", "H2O"),
    ("Speed_of_light", "approx_value", "299792458_m_per_s"),
    ("Python", "type", "interpreted_programming_language"),
    ("C++", "type", "compiled_systems_language"),
    ("CPU", "role", "executes_instructions"),
    ("GPU", "role", "parallel_matrix_and_graphics_compute"),
    ("VRAM", "role", "high_bandwidth_gpu_memory"),
    ("RAM", "role", "system_working_memory"),
    ("SSD", "role", "persistent_storage"),
    ("Transformer", "core_mechanism", "self_attention"),
    ("HDC", "core_mechanism", "hyperdimensional_vector_binding"),
    ("Eddi", "architecture", "HDC_plus_1M_mmap_context"),
]

CORE_CORPUS = """
Hello! I am Eddi, a local hyperdimensional agent.
Mathematics is the study of numbers, structures, and patterns.
Physics describes matter, energy, space, and time.
Computer science studies algorithms, data structures, and computation.
An operating system manages hardware resources and provides services for programs.
Memory hierarchy trades off capacity and speed: registers, L1/L2/L3 cache, RAM, and NVMe SSD.
"""


def teach_core_knowledge(agent: EddiAgent) -> Dict[str, int]:
    for s, r, o in CORE_FACTS:
        agent.teach(s, r, o)
    tokens = agent.ingest_text(CORE_CORPUS)
    return {"facts": len(CORE_FACTS), "tokens": tokens}


def ingest_file(agent: EddiAgent, file_path: str) -> int:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    total_tokens = 0
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    text = obj.get("text") or obj.get("content") or obj.get("prompt") or ""
                    if text:
                        total_tokens += agent.ingest_text(str(text))
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")
        total_tokens += agent.ingest_text(text)
    return total_tokens


def ingest_dir(agent: EddiAgent, dir_path: str, exts: Tuple[str, ...] = (".txt", ".md", ".jsonl")) -> int:
    total = 0
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name.endswith(exts):
                total += ingest_file(agent, os.path.join(root, name))
    return total


def save_knowledge(agent: EddiAgent, out_path: str) -> None:
    state = {
        "facts": agent.kb.facts,
        "transitions": dict(agent.transitions),
    }
    with open(out_path, "wb") as f:
        pickle.dump(state, f)


def load_knowledge(agent: EddiAgent, in_path: str) -> None:
    with open(in_path, "rb") as f:
        state = pickle.load(f)
    agent.kb.facts = state["facts"]
    for s, r, o, _ in agent.kb.facts:
        agent.kb.get_or_create(s)
        agent.kb.get_or_create(r)
        agent.kb.get_or_create(o)
    agent.transitions.update(state["transitions"])
