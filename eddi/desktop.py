"""Eddi Desktop App (Tkinter) - chat with trained memory + reasoning + language model.

Works on Windows, macOS and Linux.  If a trained brain exists it is loaded at
startup; otherwise Eddi starts fresh and can be trained from the UI.
"""

from __future__ import annotations

import sys
import threading

try:  # Tkinter is optional (some Linux distros ship Python without it)
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
except ImportError:  # pragma: no cover - platform dependent
    tk = None  # type: ignore[assignment]
    filedialog = messagebox = scrolledtext = None  # type: ignore[assignment]

from .agent import EddiAgent
from .arxiv_ingest import ingest_arxiv_snapshot
from .paths import default_brain_path
from .pipeline import ingest_file
from .training import TrainingConfig, brain_exists, load_brain, save_brain, train


class EddiDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Eddi - writable HDC memory + trained language model")
        self.root.geometry("820x600")
        self.brain_path = default_brain_path()

        if brain_exists(self.brain_path):
            self.agent = load_brain(self.brain_path)
            greeting = f"Brain loaded: {len(self.agent.kb)} facts, {self.agent.lm.vocab_size} words."
        else:
            self.agent = EddiAgent(persist_context=False)
            greeting = "No trained brain yet - press 'Train' to build one."

        self.log = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", height=24)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.entry = tk.Entry(ctrl)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.entry.bind("<Return>", lambda _: self.on_send())

        tk.Button(ctrl, text="Send", command=self.on_send).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Train", command=self.on_train).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Ingest File", command=self.on_ingest_file).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Save Brain", command=self.on_save).pack(side=tk.LEFT, padx=2)

        self._write(f"Eddi Desktop ready. {greeting}\n")
        self._write("Try: 'teach: Earth | has_moon | Moon', 'what is the moon of earth', '2+2*10'.\n")

    # --------------------------------------------------------------- helpers
    def _write(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    # --------------------------------------------------------------- actions
    def on_send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._write(f"> {text}")
        answer = self.agent.answer(text)
        self._write(f"Eddi [{answer.kind}]> {answer.text}")
        if answer.details:
            for detail in answer.details:
                self._write(f"    · {detail}")

    def on_train(self) -> None:
        self._write("Training (curriculum) ...")
        self.root.update_idletasks()

        def work() -> None:
            try:
                report = train(
                    self.agent,
                    TrainingConfig(epochs=3, save_path=self.brain_path),
                )
                self.root.after(0, lambda: self._write(report.pretty()))
            except Exception as exc:  # keep the UI alive
                message = str(exc)
                self.root.after(0, lambda: messagebox.showerror("Training failed", message))

        threading.Thread(target=work, daemon=True).start()

    def on_ingest_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select dataset (.txt, .md, .jsonl)",
            filetypes=[("Datasets", "*.txt *.md *.jsonl"), ("All files", "*.*")],
        )
        if not path:
            return
        if path.endswith(".jsonl"):
            stats = ingest_arxiv_snapshot(self.agent, path, max_papers=5000)
            self._write(f"Ingested JSONL: {stats}")
        else:
            toks = ingest_file(self.agent, path)
            self._write(f"Ingested {toks} tokens from {path}")

    def on_save(self) -> None:
        save_brain(self.agent, self.brain_path)
        self._write(f"Brain saved to {self.brain_path}")

    def on_close(self) -> None:
        try:
            save_brain(self.agent, self.brain_path)
            self._write(f"Brain saved to {self.brain_path}")
        finally:
            self.agent.close(save=False)
            self.root.destroy()


def main() -> None:
    if tk is None:  # pragma: no cover - platform dependent
        sys.exit(
            "Tkinter is not available in this Python installation.\n"
            "On Debian/Ubuntu install it with: sudo apt install python3-tk\n"
            "On Windows and macOS the standard python.org installer includes it."
        )
    root = tk.Tk()
    app = EddiDesktopApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
