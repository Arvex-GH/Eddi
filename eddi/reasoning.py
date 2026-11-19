"""Eddi's reasoning layer.

``EddiAgent`` is the public façade, but the intelligence lives here: a router
that turns a user message into one of the capabilities Eddi actually has,
instead of always falling back to n-gram continuation.

    teach / learn   -> write a fact into hyperdimensional memory
    ask / query     -> O(1) knowledge lookup with fuzzy entity resolution
    tell me about   -> summarise every fact known about a subject
    relate          -> relation discovery ("what relates Earth and Moon?")
    chain           -> multi-hop reasoning ("where does the Moon's planet orbit?")
    analogy         -> "Earth is to Sun as Moon is to ?"
    math            -> arithmetic, percentages, comparisons
    generate        -> interpolated n-gram language model
    ingest          -> feed text files/datasets into memory + model

All answers are returned as :class:`Answer` objects that say *what* was done,
so the CLI, desktop UI and tests can show the reasoning trace.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import List, Optional

_WORD_RE = re.compile(r"[A-Za-z0-9_+#']+")
_MATH_CHARS_RE = re.compile(r"^[\d\s.\+\-\*/%^()×÷,]+$")
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and",
    "what", "which", "who", "whom", "whose", "how", "why", "where", "when", "does", "do",
    "did", "tell", "me", "about", "please", "can", "you", "your", "it", "its", "that",
    "this", "with", "from", "by", "as", "be", "been", "has", "have", "had", "i", "we",
    "know", "knows", "eddi", "question", "answer", "explain", "describe", "give",
}


@dataclass
class Answer:
    """A single response plus the capability that produced it."""

    text: str
    kind: str = "generate"
    confidence: float = 0.0
    details: List[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


def _inverse_lookup(agent, subject: str, relation: Optional[str]) -> List[str]:
    """Find facts whose object (or subject) is ``subject``: ``x --rel--> subject``."""
    out: List[str] = []
    for obj, sources in agent.kb._by_o.items():
        resolved_obj = agent.resolve_entity(obj) or obj
        if resolved_obj != subject:
            continue
        for src_subject, src_relation in sources:
            if relation is not None and src_relation != relation:
                continue
            out.append(f"{src_subject} --{src_relation}--> {subject}")
    return out


def _keywords(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS]


def safe_eval_math(expression: str) -> Optional[float]:
    """Evaluate a pure arithmetic expression without ``eval``."""
    cleaned = (
        expression.replace("×", "*")
        .replace("÷", "/")
        .replace("^", "**")
        .replace(",", "")
        .strip()
        .rstrip("?")
        .strip()
    )
    if not cleaned or not _MATH_CHARS_RE.match(cleaned):
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None

    def _walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(node.op)](_walk(node.left), _walk(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(node.op)](_walk(node.operand)))
        raise ValueError("unsupported expression")

    try:
        value = _walk(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _looks_like_math(expression: str) -> bool:
    cleaned = expression.replace("×", "*").replace("÷", "/").replace("^", "**").strip().rstrip("?")
    return bool(cleaned) and bool(_MATH_CHARS_RE.match(cleaned)) and any(
        op in cleaned for op in "+-*/%"
    )


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


class Reasoner:
    """Routes a message to the right capability and composes the answer."""

    def __init__(self, agent):
        self.agent = agent

    # ------------------------------------------------------------------ main
    def answer(self, text: str) -> Answer:
        text = text.strip()
        if not text:
            return Answer("(empty message)", kind="noop")

        for handler in (
            self._handle_webpage,
            self._handle_identity,
            self._handle_math,
            self._handle_teach_rule,
            self._handle_teach_alias,
            self._handle_teach,
            self._handle_chain,
            self._handle_analogy,
            self._handle_relations,
            self._handle_ask,
            self._handle_about,
            self._handle_greeting,
        ):
            answer = handler(text)
            if answer is not None:
                return answer
        return self._handle_generate(text)

    # ------------------------------------------------------------- knowledge
    def _handle_webpage(self, text: str) -> Optional[Answer]:
        """Compose an HTML page from written components (no sampling, no training)."""
        match = re.match(
            r"^(?:make|create|build|generate|write)\s+(?:me\s+)?(?:a|an)?\s*(?:nice|good|pretty|beautiful|simple)?\s*"
            r"(landing\s+page|web\s*page|page|website|site|dashboard|article)\s*"
            r"(?:for|about|on)?\s*(.*)$",
            text.strip(), re.IGNORECASE,
        )
        if not match:
            match = re.match(r"^(?:make|create|build|generate)\s+(?:me\s+)?(?:a|an)?\s*(\w+)\s*page"
                             r"(?:\s+(?:for|about|on)\s+(.*))?$", text.strip(), re.IGNORECASE)
        if not match:
            return None
        kind = (match.group(1) or "").lower()
        topic = (match.group(2) or "").strip()
        request = f"{kind} for {topic}" if topic else f"{kind} page"
        from .html_library import HtmlBuilder, spec_from_request

        spec = spec_from_request(request)
        if kind == "article":
            spec.layout = "article"
        elif kind == "dashboard":
            spec.layout = "dashboard"
        page = HtmlBuilder().build(spec)
        return Answer(
            page,
            kind="webpage",
            confidence=1.0,
            details=[
                f"layout={spec.layout}",
                "composed from written components; the language model was not used",
            ],
        )

    def _handle_teach_rule(self, text: str) -> Optional[Answer]:
        match = re.match(r"^(?:rule|when)\s*[:\-]?\s*(.+?)\s*(?:=>|->|implies)\s*(.+?)[\.]*$",
                         text, re.IGNORECASE)
        if not match:
            return None
        premise, conclusion = (g.strip() for g in match.groups())
        p, c = self.agent.teach_rule(premise, conclusion)
        derived = len(self.agent.kb.derived)
        return Answer(
            f"Rule written: {p} => {c}  ({derived} fact(s) now derived, 0 training tokens)",
            kind="rule",
            confidence=1.0,
            details=["writable inference: forward chaining over existing facts"],
        )

    def _handle_teach_alias(self, text: str) -> Optional[Answer]:
        match = re.match(r"^(?:alias|also\s+called)\s*[:\-]?\s*(.+?)\s*(?:=|->|means|is)\s*(.+?)[\.]*$",
                         text, re.IGNORECASE)
        if not match:
            return None
        alias, canonical = (g.strip() for g in match.groups())
        resolved = self.agent.teach_alias(alias, canonical)
        return Answer(
            f"Alias written: '{alias}' -> {resolved}",
            kind="alias",
            confidence=1.0,
            details=["writable vocabulary: no retraining needed"],
        )

    def _handle_teach(self, text: str) -> Optional[Answer]:
        match = re.match(r"^(?:teach|learn|remember)\s*[:\-]?\s*(.+)$", text, re.IGNORECASE)
        if not match:
            return None
        body = match.group(1)
        if "|" in body:
            parts = [p.strip() for p in body.split("|")]
            if len(parts) != 3:
                return Answer("Use: teach subject | relation | object", kind="help")
            s, r, o = parts
        else:
            patterns = [
                r"^(?:that\s+)?(.+?)\s+(?:'s|’s)\s+([\w \-]+?)\s+(?:is|are)\s+(.+)$",
                r"^(?:that\s+)?(.+?)\s+([\w \-]+?)\s+(?:is|are|=)\s+(.+)$",
            ]
            parsed = None
            for pattern in patterns:
                m = re.match(pattern, body, re.IGNORECASE)
                if m:
                    parsed = tuple(g.strip() for g in m.groups())
                    break
            if parsed is None:
                return Answer(
                    "Use: teach subject | relation | object  (or 'teach X has relation Y')",
                    kind="help",
                )
            s, r, o = parsed
        created = self.agent.teach(s, r, o)
        extra = f" (+{created - 1} derived by rules)" if created > 1 else ""
        return Answer(f"Learned: ({s}) --{r}--> ({o}){extra}", kind="teach", confidence=1.0)

    def _handle_ask(self, text: str) -> Optional[Answer]:
        match = re.match(r"^(?:ask|query)\s*[:\-]?\s*(.+)$", text, re.IGNORECASE)
        if match:
            body = match.group(1)
            if "|" in body:
                parts = [p.strip() for p in body.split("|")]
                if len(parts) == 2:
                    return self._answer_relation(parts[0], parts[1], original=text)
            if not body.strip():
                return Answer("Use: ask <subject> | <relation>", kind="help")
            tokens = _WORD_RE.findall(body)
            if len(tokens) >= 2:
                subject = self.agent.resolve_entity(" ".join(tokens[:-1]))
                relation = self.agent.resolve_relation(tokens[-1])
                if subject and relation:
                    return self._answer_relation(subject, relation, original=text)
            return Answer("Use: ask Eddi | uses", kind="help")

        # "what is the <relation> of <subject>"
        match = re.match(r"^what\s+(?:is|are|was)\s+(?:the\s+)?(.+?)\s+of\s+(?:the\s+)?(.+?)[\?\.\!]*$",
                         text, re.IGNORECASE)
        if match:
            return self._answer_attribute(match.group(1), match.group(2))

        # "what does <subject> <verb>?"  /  "who does <subject> <verb>?"
        match = re.match(r"^(?:what|who)\s+(?:does|do|did)\s+(?:the\s+)?(.+?)\s+([\w \-]+?)[\?\.\!]*$",
                         text, re.IGNORECASE)
        if match:
            subject, verb = match.group(1), match.group(2)
            relation = self.agent.resolve_relation(verb)
            resolved = self.agent.resolve_entity(subject)
            if resolved and relation:
                return self._answer_relation(resolved, relation, original=text)
            return None

        # "what is <subject>" - answer from memory if anything is known
        match = re.match(r"^what(?:'s|’s|\s+is|\s+are)\s+(?:the\s+)?(.+?)[\?\.\!]*$",
                         text, re.IGNORECASE)
        if match:
            phrase = match.group(1)
            subject = self.agent.resolve_entity(phrase)
            relation = self.agent.resolve_relation(phrase)
            if relation is not None and not self.agent.kb.facts_about(subject or ""):
                return Answer(
                    f"I know the relation '{relation}' but you need to name a subject, "
                    f"e.g. 'what is the {relation} of Earth'.",
                    kind="help",
                )
            if subject and self.agent.kb.facts_about(subject):
                return self._about(subject)
            return None
        return None

    def _answer_attribute(self, relation_text: str, subject_text: str) -> Answer:
        """Answer 'what is the <relation_text> of <subject_text>'."""
        # A written alias may cover the whole object phrase ("the red planet").
        aliased = self.agent.kb.resolve(subject_text)
        if aliased and self.agent.kb.facts_about(aliased):
            relation = self.agent.resolve_relation(relation_text)
            if relation and self.agent.kb.query(aliased, relation):
                return self._answer_relation(aliased, relation)
        # The whole phrase may itself be a known entity ("speed of light").
        phrase = self.agent.kb.resolve(f"{relation_text} of {subject_text}")
        if phrase and self.agent.kb.facts_about(phrase):
            return self._about(phrase)

        subject = self.agent.resolve_entity(subject_text)
        relation = self.agent.resolve_relation(relation_text)
        if subject is None:
            return Answer(
                f"I don't know '{subject_text}' yet. Try 'teach {subject_text} <relation> <value>'.",
                kind="unknown",
            )
        if relation and self.agent.kb.query(subject, relation):
            return self._answer_relation(subject, relation)

        # Inverse relations: "what is the capital of Germany?" -> (Berlin, capital_of, Germany)
        inverse = _inverse_lookup(self.agent, subject, relation) if relation else []
        if inverse:
            return Answer(
                "; ".join(inverse),
                kind="ask",
                confidence=1.0,
                details=["inverse relation lookup (object -> subject)"],
            )

        # The "relation" may actually be a known entity: "what is the moon of earth"
        object_entity = self.agent.resolve_entity(relation_text)
        if object_entity:
            hits = [
                f"{subject} --{rel}--> {obj}"
                for rel, obj, _conf in self.agent.kb.facts_about(subject)
                if (self.agent.resolve_entity(obj) or obj) == object_entity
            ]
            hits.extend(self.agent.relate(subject, object_entity))
            if hits:
                unique = list(dict.fromkeys(hits))
                return Answer("; ".join(unique), kind="ask", confidence=1.0,
                              details=["reverse lookup: entity matched, relation inferred"])

        about = self._about(subject)
        return Answer(
            f"I don't know the '{relation_text}' of {subject} yet.\nWhat I do know:\n{about.text}",
            kind="unknown",
            details=["attribute not found"],
        )

    def _answer_relation(self, subject: str, relation: str, original: str = "") -> Answer:
        resolved_subject = self.agent.resolve_entity(subject) or subject
        resolved_relation = self.agent.resolve_relation(relation) or relation
        hits = self.agent.kb.query(resolved_subject, resolved_relation, top_k=3)
        if not hits:
            return Answer(
                f"I don't know that yet. Try 'teach {resolved_subject} {resolved_relation} <value>'.",
                kind="unknown",
            )
        objects = ", ".join(obj for obj, _ in hits)
        details = [f"memory: exact O(1) lookup on ({resolved_subject}, {resolved_relation})"]
        robust = self.agent.kb.probe(resolved_subject, resolved_relation, top_k=3)
        if robust and len(robust) > 1:
            details.append("HDC probe: " + ", ".join(f"{o} ({s:.2f})" for o, s in robust if s > 0.55))
        return Answer(
            f"{resolved_subject} --{resolved_relation}--> {objects}",
            kind="ask",
            confidence=hits[0][1],
            details=details,
        )

    def _about(self, subject: str) -> Answer:
        facts = self.agent.kb.facts_about(subject)
        if not facts:
            return Answer(f"I don't know anything about '{subject}' yet.", kind="unknown")
        lines = [f"{subject}:"]
        for relation, obj, _conf in facts[:12]:
            lines.append(f"  {relation}: {obj}")
        return Answer("\n".join(lines), kind="about", confidence=1.0)

    def _handle_about(self, text: str) -> Optional[Answer]:
        if re.match(r"^(?:tell\s+me\s+about|facts\s+about|what\s+do\s+you\s+know\s+about|explain)\s*[:\-]?\s*[\?\.\!]*$",
                    text, re.IGNORECASE):
            return Answer("Tell me about what? e.g. 'tell me about Earth'.", kind="help")
        match = re.match(
            r"^(?:tell\s+me\s+about|facts\s+about|what\s+do\s+you\s+know\s+about|explain)\s*[:\-]?\s*(.+?)[\?\.\!]*$",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        subject = self.agent.resolve_entity(match.group(1))
        if subject is None:
            return Answer(f"I don't know anything about '{match.group(1)}' yet.", kind="unknown")
        return self._about(subject)

    def _handle_relations(self, text: str) -> Optional[Answer]:
        match = re.match(
            r"^(?:how\s+(?:is|are)\s+(.+?)\s+related\s+to\s+(.+?)|what\s+relates\s+(.+?)\s+and\s+(.+?))[\?\.\!]*$",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        groups = [g for g in match.groups() if g]
        a, b = groups[0], groups[1]
        relations = self.agent.relate(a, b)
        if not relations:
            return Answer(f"I don't know any relation between '{a}' and '{b}'.", kind="unknown")
        return Answer(
            "; ".join(relations),
            kind="relate",
            confidence=1.0,
            details=["searched relations from both directions"],
        )

    def _handle_chain(self, text: str) -> Optional[Answer]:
        match = re.match(r"^(?:chain|then)\s*[:\-]?\s*(.+)$", text, re.IGNORECASE)
        if match:
            body = match.group(1)
            parts = [p.strip() for p in body.split("|")]
            if len(parts) < 2:
                return Answer("Use: chain start | rel1 | rel2", kind="help")
            results = self.agent.chain(parts[0], parts[1:])
            if not results:
                return Answer(f"No chain found from '{parts[0]}'.", kind="unknown")
            return Answer(f"{parts[0]} -> {', '.join(results)}", kind="chain", confidence=1.0,
                          details=["multi-hop traversal"])
        match = re.match(
            r"^(?:what|where|who)\s+(.+?)\s+(?:of\s+)?(?:the\s+)?([\w \-]+?)['’]s\s+([\w \-]+?)[\?\.\!]*$",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        relation_out, start, relation_in = (g.strip() for g in match.groups())
        results = self.agent.chain(start, [relation_in, relation_out])
        if not results:
            return None
        return Answer(
            f"{start} --{relation_in}--> --{relation_out}--> {', '.join(results)}",
            kind="chain",
            confidence=1.0,
            details=["two-hop traversal"],
        )

    def _handle_analogy(self, text: str) -> Optional[Answer]:
        match = re.match(
            r"^(.+?)\s*(?:is\s+to|:)\s*(.+?)\s*(?:as|::)\s*(.+?)\s*(?:is\s+to|:)\s*(.+?)[\?\.\!]*$",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        a, b, c, _ = (g.strip() for g in match.groups())
        results = self.agent.kb.analogy(a, b, c)
        if not results:
            return Answer("I can't build that analogy from what I know yet.", kind="unknown")
        relation, obj, score = results[0]
        return Answer(
            f"{c} --{relation}--> {obj}  (by analogy with {a} --{relation}--> {b}, score {score:.2f})",
            kind="analogy",
            confidence=score,
        )


    def _handle_identity(self, text: str) -> Optional[Answer]:
        """Answer questions about Eddi himself from written memory (no guessing)."""
        lowered = " " + re.sub(r"[^a-z ]+", " ", text.lower()) + " "
        identity_markers = (
            " your name ", " your identity ", " who are you ", " what are you ",
            " tell me about yourself ", " introduce yourself ", " who is eddi ",
            " what is eddi ", " about yourself ", " your purpose ", " what do you do ",
        )
        if not any(marker in lowered for marker in identity_markers):
            return None

        name = self.agent.resolve_entity("eddi") or "Eddi"
        name_hits = self.agent.kb.query(name, "name", top_k=1)
        facts = self.agent.kb.facts_about(name)
        if not facts:
            return Answer(
                "I'm Eddi. I don't have any facts written about me yet - "
                "try 'teach Eddi | purpose | ...'.",
                kind="identity",
                confidence=1.0,
            )
        lines = [f"My name is {name_hits[0][0] if name_hits else name}."]
        for relation, obj, _conf in facts:
            if relation == "name":
                continue
            lines.append(f"  {relation.replace('_', ' ')}: {obj.replace('_', ' ')}")
        return Answer(
            "\n".join(lines),
            kind="identity",
            confidence=1.0,
            details=["answered from written memory, not from the language model"],
        )

    # ------------------------------------------------------------------- math
    def _handle_math(self, text: str) -> Optional[Answer]:
        match = re.match(r"^(?:calc|compute|what\s+is|what's|solve)?\s*[:\-]?\s*(.+)$", text, re.IGNORECASE)
        if not match:
            return None
        expr = match.group(1).strip()
        if not _MATH_CHARS_RE.match(expr.rstrip("?").strip()):
            return None
        value = safe_eval_math(expr)
        if value is None:
            if _looks_like_math(expr):
                return Answer(
                    f"I can't evaluate `{expr}` (division by zero, or the result is not a finite number).",
                    kind="math_error",
                )
            return None
        return Answer(f"{_format_number(value)}", kind="math", confidence=1.0, details=[f"evaluated `{expr}`"])

    # ------------------------------------------------------------- fallbacks
    def _handle_greeting(self, text: str) -> Optional[Answer]:
        lowered = text.lower().strip("!.? ")
        greetings = {"hi", "hello", "hey", "yo", "hallo", "servus", "moin"}
        if lowered in greetings:
            return Answer(
                "Hi! I'm Eddi. I remember facts, reason over them, do math and write text. "
                "Try: teach Earth has_moon Moon | what is the moon of earth | Earth is to Sun as Moon is to ?",
                kind="greeting",
            )
        if lowered in {"help", "commands", "?"}:
            return Answer(
                "Commands:\n"
                "  teach <subject> | <relation> | <object>\n"
                "  teach <subject> has <relation> <object>\n"
                "  ask <subject> | <relation>\n"
                "  what is the <relation> of <subject>\n"
                "  tell me about <subject>\n"
                "  how is <a> related to <b>\n"
                "  chain <start> | <rel1> | <rel2>\n"
                "  rule <rel1> => <rel2>      (writable inference, no training)\n"
                "  alias <phrase> = <canonical>\n"
                "  <a> is to <b> as <c> is to ?\n"
                "  2 + 2 * 10\n"
                "  anything else -> language-model continuation",
                kind="help",
            )
        return None

    def _handle_generate(self, text: str) -> Answer:
        if self.agent.lm.trained_tokens == 0:
            return Answer(
                "I have no trained language model yet. Ingest a text file to teach me, "
                "or run `python -m eddi.cli --train` to build the built-in curriculum.",
                kind="unknown",
            )
        generated = self.agent.generate(text, max_tokens=28)
        if not generated:
            return Answer("I have nothing to build on yet.", kind="unknown")
        return Answer(generated, kind="generate", confidence=0.5, details=["interpolated n-gram model"])
