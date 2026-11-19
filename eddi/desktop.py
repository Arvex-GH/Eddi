"""Eddi Desktop App (Tkinter) — Chat, Teach, Synthesize, and Ingest Datasets."""

import tkinter as tk
from tkinter import filedialog, scrolledtext
from .agent import EddiAgent
from .arxiv_ingest import ingest_arxiv_snapshot
from .pipeline import ingest_file, load_knowledge, save_knowledge, teach_core_knowledge
from .synthesizer import CompositionalSynthesizer, teach_synthesis_primitives


class EddiDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Eddi — Writable HDC + 1M-Token Context Agent")
        self.root.geometry("760x560")

        self.agent = EddiAgent()
        teach_core_knowledge(self.agent)
        self.synth = CompositionalSynthesizer(self.agent)
        teach_synthesis_primitives(self.synth)

        self.log = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", height=20)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.entry = tk.Entry(ctrl)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.entry.bind("<Return>", lambda _: self.on_send())

        tk.Button(ctrl, text="Ask / Generate", command=self.on_send).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Ingest File", command=self.on_ingest_file).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Save KB", command=self.on_save).pack(side=tk.LEFT, padx=2)

        self._write("Eddi Desktop ready. Type a prompt, or 'teach: A | rel | B', or 'ask: A | rel'.\n")

    def _write(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def on_send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._write(f"> {text}")

        if text.startswith("teach:"):
            parts = [p.strip() for p in text[6:].split("|")]
            if len(parts) == 3:
                self.agent.teach(parts[0], parts[1], parts[2])
                self._write(f"Learned: ({parts[0]}, {parts[1]}, {parts[2]})")
            else:
                self._write("Format: teach: subject | relation | object")
        elif text.startswith("ask:"):
            parts = [p.strip() for p in text[4:].split("|")]
            if len(parts) == 2:
                self._write("Answer: " + str(self.agent.ask(parts[0], parts[1])))
            else:
                self._write("Format: ask: subject | relation")
        else:
            out = self.agent.generate(text, max_tokens=25)
            self._write(out if out else "(No sequence match yet — ingest a file or teach facts!)")

    def on_ingest_file(self) -> None:
        path = filedialog.askopenfilename(title="Select dataset (.txt, .md, .jsonl)")
        if not path:
            return
        if path.endswith(".jsonl"):
            stats = ingest_arxiv_snapshot(self.agent, path, max_papers=5000)
            self._write(f"Ingested JSONL: {stats}")
        else:
            toks = ingest_file(self.agent, path)
            self._write(f"Ingested {toks} tokens from {path}")

    def on_save(self) -> None:
        save_knowledge(self.agent, "eddi_knowledge.pkl")
        self._write("Saved knowledge to eddi_knowledge.pkl")


def main() -> None:
    root = tk.Tk()
    app = EddiDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
