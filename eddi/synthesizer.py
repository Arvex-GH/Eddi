"""Compositional Program Synthesizer for Eddi.

Assembles new programs from typed primitives and composition rules stored in
Eddi's WritableKnowledgeBase, rather than echoing verbatim n-grams.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from .agent import EddiAgent


@dataclass
class CodePrimitive:
    name: str
    kind: str       # e.g., "container", "widget", "action"
    language: str   # e.g., "html", "csharp"
    template: str   # parameterized template with {slots}


class CompositionalSynthesizer:
    def __init__(self, agent: EddiAgent):
        self.agent = agent
        self.primitives: Dict[str, CodePrimitive] = {}

    def register_primitive(self, prim: CodePrimitive) -> None:
        key = f"{prim.language}:{prim.name}"
        self.primitives[key] = prim
        self.agent.teach(prim.name, "kind", prim.kind)
        self.agent.teach(prim.name, "language", prim.language)

    def synthesize(self, language: str, widget: str, label: str, message: str) -> str:
        """Compose a new program from separate primitives in memory."""
        container = self.primitives.get(f"{language}:container")
        comp = self.primitives.get(f"{language}:{widget}")
        action = self.primitives.get(f"{language}:alert_action")

        if not (container and comp and action):
            raise ValueError(f"Missing primitives for {language}/{widget}")

        rendered_action = action.template.format(message=message)
        rendered_widget = comp.template.format(label=label, action=rendered_action)
        return container.template.format(body=rendered_widget)


def teach_synthesis_primitives(synth: CompositionalSynthesizer) -> None:
    # HTML primitives
    synth.register_primitive(CodePrimitive("container", "container", "html", "<!DOCTYPE html><html><body>\n  {body}\n</body></html>"))
    synth.register_primitive(CodePrimitive("button", "widget", "html", '<button onclick="{action}">{label}</button>'))
    synth.register_primitive(CodePrimitive("alert_action", "action", "html", "alert('{message}')"))

    # C# WinForms/Console-style primitives
    synth.register_primitive(CodePrimitive("container", "container", "csharp", "using System;\nclass App {{\n  static void Main() {{\n    {body}\n  }}\n}}"))
    synth.register_primitive(CodePrimitive("button", "widget", "csharp", 'var btn = new Button {{ Text = "{label}" }};\n    btn.Click += (s, e) => {action};'))
    synth.register_primitive(CodePrimitive("alert_action", "action", "csharp", 'Console.WriteLine("{message}");'))
