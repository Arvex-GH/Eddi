"""arXiv Snapshot Ingestion Adapter for Eddi.

Streams arXiv OAI-PMH JSONL metadata (e.g. arxiv-metadata-oai-snapshot.json)
into Eddi's 1M-token disk-backed mmap context and WritableKnowledgeBase.
"""

import json
from pathlib import Path
from typing import Optional
from .agent import EddiAgent


def ingest_arxiv_snapshot(agent: EddiAgent, jsonl_path: str, max_papers: Optional[int] = None) -> dict:
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"arXiv snapshot not found: {jsonl_path}")

    papers = 0
    tokens = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if max_papers is not None and papers >= max_papers:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            title = (obj.get("title") or "").replace("\n", " ").strip()
            abstract = (obj.get("abstract") or "").replace("\n", " ").strip()
            arxiv_id = obj.get("id") or f"paper_{papers}"
            categories = obj.get("categories") or "unknown"

            if title:
                agent.teach(arxiv_id, "title", title[:120])
                agent.teach(arxiv_id, "category", categories)
                tokens += agent.ingest_text(f"{title} {abstract}")
                papers += 1

    return {"papers": papers, "tokens": tokens}
