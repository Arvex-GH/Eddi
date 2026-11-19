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
import datetime
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
            self._handle_tell_about_self,
            self._handle_ask_about_self,
            self._handle_birth_year,
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

    # The user has a stable identity in memory, so first-person statements are
    # knowledge, not questions. Eddi previously treated "I am 16 years old" as a
    # lookup for the word "years" and answered "I don't know".
    USER_SUBJECT = "user"

    # Told-back phrasing, not bare nouns: "you live in Frankfurt" reads like
    # English, "your lives in is Frankfurt" does not.
    _RELATION_LABEL = {
        "age": ("your age is", "your age"),
        "name": ("your name is", "your name"),
        "occupation": ("your job is", "your job"),
        "lives_in": ("you live in", "where you live"),
        "likes": ("you like", "what you like"),
        "work": ("you work as", "your work"),
    }

    _SELF_TELL = [
        # "I am 16 years old"            -> age
        (re.compile(r"^\s*i\s*am\s+(\d+)\s*(?:years?|yrs?)\s*old\s*[.!]?\s*$", re.I), "age"),
        # "I am a student" / "I'm an engineer"
        (re.compile(r"^\s*i\s*(?:'m|\s+am)\s+(?:a|an)\s+(.+?)\s*[.!]?\s*$", re.I), "occupation"),
        # "I live in Frankfurt"
        (re.compile(r"^\s*i\s+live\s+in\s+(.+?)\s*[.!]?\s*$", re.I), "lives_in"),
        # "I like pizza" / "I love Rust"
        (re.compile(r"^\s*i\s+(?:like|love|enjoy|prefer)\s+(.+?)\s*[.!]?\s*$", re.I), "likes"),
        # "I work as a nurse" / "I work at Acme"
        (re.compile(r"^\s*i\s+work\s+(?:as|at)\s+(?:a\s+|an\s+)?(.+?)\s*[.!]?\s*$", re.I), "work"),
        # "my name is Alex" / "my favourite colour is blue"
        (re.compile(r"^\s*my\s+name\s+is\s+(.+?)\s*[.!]?\s*$", re.I), "name"),
        (re.compile(r"^\s*my\s+favou?rite\s+(\w+)\s+is\s+(.+?)\s*[.!]?\s*$", re.I), None),
        # "I am Alex" (bare, no article) - name, only if it looks like a name
        (re.compile(r"^\s*[Ii]\s*(?:'m|\s+am)\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)\s*[.!]?\s*$"), "name"),
    ]

    def _handle_tell_about_self(self, text: str) -> Optional[Answer]:
        """Remember a fact the user states about themselves.

        Statements, not questions: "I am 16 years old" is something to keep, and
        answering it with a knowledge lookup was a category error.
        """
        for pattern, relation in self._SELF_TELL:
            match = pattern.match(text)
            if not match:
                continue
            if relation is None:
                # "my favourite colour is blue" -> favourite_colour
                relation = "favourite_" + match.group(1).lower()
                obj = match.group(2)
                label = "your favourite " + match.group(1).lower() + " is"
            else:
                obj = match.group(1)
                told, _ = self._RELATION_LABEL.get(
                    relation, (f"your {relation.replace('_', ' ')} is", relation)
                )
                label = told
            obj = obj.strip(" .!")
            if not obj:
                continue
            # Speaking a fact twice should not duplicate it.
            existing = [o for o, _ in self.agent.kb.query(self.USER_SUBJECT, relation)]
            if any(obj.lower() == e.lower() for e in existing):
                return Answer(
                    f"I already knew that {label} {obj}.",
                    kind="remember",
                    confidence=1.0,
                )
            self.agent.teach(self.USER_SUBJECT, relation, obj)
            return Answer(
                f"Got it - I'll remember that {label} {obj}.",
                kind="remember",
                confidence=1.0,
                details=[f"memory: ({self.USER_SUBJECT}, {relation}, {obj})"],
            )
        return None

    _SELF_ASK = [
        (re.compile(r"\bhow\s+old\s+am\s+i\b|\bwhat(?:'s|\s+is)\s+my\s+age\b", re.I), "age"),
        (re.compile(r"\bwhat(?:'s|\s+is)\s+my\s+name\b|\bwho\s+am\s+i\b", re.I), "name"),
        (re.compile(r"\bwhere\s+do\s+i\s+live\b", re.I), "lives_in"),
        (re.compile(r"\bwhat\s+do\s+i\s+do\b|\bwhat(?:'s|\s+is)\s+my\s+(?:job|occupation)\b", re.I), "occupation"),
        (re.compile(r"\bwhat\s+do\s+i\s+like\b", re.I), "likes"),
        # "what is my favourite colour" -> favourite_colour
        (re.compile(r"\bwhat(?:'s|\s+is)\s+my\s+favou?rite\s+(\w+)\b", re.I), "favourite_*"),
    ]

    # How to say each fact back to the user.
    _ANSWER_PHRASE = {
        "age": lambda v: f"You're {v}.",
        "name": lambda v: f"Your name is {v}.",
        "lives_in": lambda v: f"You live in {v}.",
        "occupation": lambda v: f"You're a {v}.",
        "likes": lambda v: f"You like {v}.",
        "work": lambda v: f"You work as {v}.",
    }

    def _handle_ask_about_self(self, text: str) -> Optional[Answer]:
        """Answer questions about the user from the facts they told Eddi.

        Remembering would be pointless if it could not be recalled, so this is
        the other half of _handle_tell_about_self.
        """
        for pattern, relation in self._SELF_ASK:
            match = pattern.search(text)
            if not match:
                continue
            if relation == "favourite_*":
                thing = match.group(1).lower()
                relation = "favourite_" + thing
                unknown_msg = f"I don't know your favourite {thing} yet - tell me and I'll remember."
                answer_phrase = lambda v, _t=thing: f"Your favourite {_t} is {v}."
            else:
                _, topic = self._RELATION_LABEL.get(
                    relation, ("", f"your {relation.replace('_', ' ')}")
                )
                unknown_msg = f"I don't know {topic} yet - tell me and I'll remember."
                answer_phrase = self._ANSWER_PHRASE.get(
                    relation, lambda v: f"Your {relation.replace('_', ' ')} is {v}."
                )

            hits = [o for o, _ in self.agent.kb.query(self.USER_SUBJECT, relation)]
            if not hits:
                return Answer(
                    unknown_msg,
                    kind="unknown",
                    details=[f"no fact stored for ({self.USER_SUBJECT}, {relation})"],
                )
            return Answer(
                answer_phrase(hits[0]),
                kind="about_self",
                confidence=1.0,
                details=[f"memory: ({self.USER_SUBJECT}, {relation})"],
            )
        return None

    # Questions whose answer is *computed* from a stored fact plus the current
    # date, not looked up. Kept separate from _handle_ask because the value was
    # never written down - deriving it is the whole point.
    _BIRTH_YEAR_Q = re.compile(
        r"\b(?:which|what)\s+(?:year|date)\s+(?:was|am)\s+i\s+born"
        r"|\bwhen\s+was\s+i\s+born"
        r"|\byear\s+(?:of\s+my\s+)?birth\b"
        r"|\bborn\s+in\s+which\s+year",
        re.IGNORECASE,
    )
    # "if I am 16" / "I am 16" inside a question - an explicit value to use.
    _INLINE_AGE = re.compile(r"\b(?:if\s+)?i(?:'m|\s+am)\s+(\d{1,3})\b", re.IGNORECASE)

    def _handle_birth_year(self, text: str) -> Optional[Answer]:
        """Compute a birth year from an age and the current date.

        The age may be stated in the question ("if I am 16 ...") or already known
        from something the user told Eddi. The result is a range, not a single
        year, because an age does not fix a birthday: someone who is 16 today was
        born in the current year minus 16, or minus 17 if their birthday has not
        yet passed.
        """
        if not self._BIRTH_YEAR_Q.search(text):
            return None

        source = ""
        inline = self._INLINE_AGE.search(text)
        if inline:
            age = int(inline.group(1))
            source = f"the age of {age} in your question"
        else:
            stored = [o for o, _ in self.agent.kb.query(self.USER_SUBJECT, "age")]
            digits = [o for o in stored if str(o).strip().isdigit()]
            if not digits:
                return Answer(
                    "I don't know your age yet - tell me ('I am 16 years old') and ask again.",
                    kind="unknown",
                    details=["no fact stored for (user, age)"],
                )
            age = int(str(digits[0]).strip())
            source = f"your age, {age}, which you told me"

        if not 0 <= age <= 130:
            return Answer(
                f"{age} is not a plausible age, so I can't work out a birth year from it.",
                kind="math_error",
            )

        this_year = datetime.date.today().year
        latest = this_year - age          # birthday already passed this year
        earliest = this_year - age - 1    # birthday still to come
        return Answer(
            f"You were born in {earliest} or {latest}.\n"
            f"  working: {this_year} - {age} = {latest}. "
            f"Because an age alone does not fix a birthday, someone who is {age} "
            f"today could have been born in {earliest} (birthday not yet reached) "
            f"or {latest} (birthday already passed).",
            kind="derive",
            confidence=1.0,
            details=[
                f"computed from {source} and today's date ({this_year})",
                "no fact for a birth year was stored; this is derived, not remembered",
            ],
        )

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

        # "what <relation> is <subject>" - the way people actually ask
        # ("what colour is zorblax?"). The relation is whatever sits between
        # 'what' and 'is', and it only routes when that word really is a
        # written relation, so ordinary questions fall through untouched.
        match = re.match(r"^what\s+(.+?)\s+(?:is|are|was)\s+(?:the\s+)?(.+?)[\?\.\!]*$",
                         text, re.IGNORECASE)
        if match:
            relation = self.agent.resolve_relation(match.group(1))
            subject = self.agent.resolve_entity(match.group(2))
            if relation and subject and self.agent.kb.query(subject, relation):
                return self._answer_relation(subject, relation, original=text)

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
                f"I don't know '{subject_text}' yet. Teach me a fact about it and ask again.",
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
                f"I don't know that yet. Teach me a {resolved_relation} for {resolved_subject} and ask again.",
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
    # Words that carry no question on their own. Used to decide whether a
    # message is *only* a pleasantry, so "hi, what is python?" still routes to
    # the knowledge handlers while "hi how are you?" gets a greeting.
    _CHITCHAT_WORDS = frozenset(
        """
        hi hello hey yo hallo servus moin hola ciao greetings howdy
        good morning afternoon evening night day
        how are is it you you're your doing going there thanks thank
        appreciate cheers bye goodbye cya later see soon ok okay
        eddi please and so too very much really
        """.split()
    )

    _GREETING_OPENERS = re.compile(
        r"^\s*(?:hi|hello|hey|yo|hallo|servus|moin|hola|ciao|howdy|greetings"
        r"|good\s+(?:morning|afternoon|evening|day))\b",
        re.IGNORECASE,
    )
    _THANKS = re.compile(r"^\s*(?:thanks|thank\s+you|cheers|ta)\b", re.IGNORECASE)
    _FAREWELL = re.compile(
        r"^\s*(?:bye|goodbye|good\s+bye|see\s+(?:you|ya)|cya|later|good\s+night)\b",
        re.IGNORECASE,
    )
    _HOW_ARE_YOU = re.compile(
        r"\b(?:how\s+are\s+(?:you|u)|how'?s\s+it\s+going|how\s+you\s+doing"
        r"|how\s+do\s+you\s+(?:do|feel)|are\s+you\s+(?:ok|okay|well|alright))\b",
        re.IGNORECASE,
    )

    def _is_small_talk(self, text: str) -> bool:
        """True when nothing in the message could be a question about knowledge.

        A message is small talk if it opens with a greeting/thanks/farewell and
        every remaining word is itself chit-chat. That keeps "hi" and
        "hi how are you?" together while letting "hi, what is python?" through
        to the knowledge handlers, where the real question lives.
        """
        words = [w for w in re.findall(r"[a-z']+", text.lower())]
        if not words:
            return False
        # "how are you?" has no greeting word, so the opener test alone misses it.
        if not (
            self._GREETING_OPENERS.search(text)
            or self._THANKS.search(text)
            or self._FAREWELL.search(text)
            or self._HOW_ARE_YOU.search(text)
        ):
            return False
        return all(w in self._CHITCHAT_WORDS for w in words)

    def _handle_greeting(self, text: str) -> Optional[Answer]:
        if not self._is_small_talk(text):
            return None

        if self._THANKS.search(text):
            return Answer(
                "Any time. Ask me something else whenever you like.",
                kind="greeting",
            )
        if self._FAREWELL.search(text):
            return Answer(
                "See you. I keep what you taught me between sessions.",
                kind="greeting",
            )
        if self._HOW_ARE_YOU.search(text):
            return Answer(
                "I'm doing well, thanks for asking - I'm a small program, so my mood "
                "is mostly intact memory. What can I help with?",
                kind="greeting",
            )
        return Answer(
            "Hi! I'm Eddi. I remember facts, reason over them, do math and write text. "
            "Try: teach Earth has_moon Moon | what is the moon of earth | Earth is to Sun as Moon is to ?",
            kind="greeting",
        )

    def _handle_generate(self, text: str) -> Answer:
        if self.agent.lm.trained_tokens == 0:
            return Answer(
                "I have no trained language model yet. Ingest a text file to teach me, "
                "or run `python -m eddi.cli --train` to build the built-in curriculum.",
                kind="unknown",
            )
        # An n-gram model will always continue *something*, so a fluent answer is
        # not evidence of knowledge. If the question is about a word the model has
        # never seen, the continuation cannot be about it - refusing is correct,
        # and returning a confident-looking sentence would be a lie.
        #
        # Never refuse something the knowledge base can actually answer: facts
        # written with `teach` are not in the language model's vocabulary, and a
        # guard placed before this check would deny knowledge Eddi really has.
        if self._kb_can_answer(text):
            return self._answer_from_kb(text)
        unknown = self._unknown_key_terms(text)
        if unknown:
            return Answer(
                f"I don't know anything about '{unknown[0]}' yet.",
                kind="unknown",
                details=[f"not in vocabulary: {', '.join(unknown)}"],
            )
        generated = self.agent.generate(text, max_tokens=28)
        if not generated:
            return Answer("I have nothing to build on yet.", kind="unknown")
        # An n-gram model primes on its own prompt, so the cheapest continuation
        # is to repeat the question. That is not an answer, so strip it.
        body = generated.strip()
        prompt = text.strip()
        if body.lower().startswith(prompt.lower()):
            body = body[len(prompt):].lstrip(" .,:;-").strip()
        if len(body.split()) < 3:
            return Answer(
                f"I don't have anything specific about '{text.strip()}' - "
                "my language model has too little text to build a real answer.",
                kind="unknown",
            )
        return Answer(
            body,
            kind="generate",
            confidence=0.5,
            details=["interpolated n-gram model"],
        )

    def _kb_can_answer(self, text: str) -> bool:
        """Is there a written fact that plausibly answers this question?"""
        subject = self.agent.resolve_entity(text)
        if subject and self.agent.kb.facts_about(subject):
            return True
        for word in _WORD_RE.findall(text):
            resolved = self.agent.resolve_entity(word)
            if resolved and self.agent.kb.facts_about(resolved):
                return True
        return False

    def _answer_from_kb(self, text: str) -> Answer:
        """Answer from written facts after the phrase parsers declined."""
        subject = self.agent.resolve_entity(text)
        if subject and self.agent.kb.facts_about(subject):
            return self._about(subject)
        for word in _WORD_RE.findall(text):
            resolved = self.agent.resolve_entity(word)
            if resolved and self.agent.kb.facts_about(resolved):
                return self._about(resolved)
        return Answer(f"I don't know anything about '{text.strip()}' yet.", kind="unknown")

    _STOP_TERMS = frozenset(
        """
        what is are was were the a an of for to in on at about tell me you your
        i we they it this that these those do does did can could should would
        please and or but how why when where which who whom whose with from by
        know more please explain describe give show make made making get got
        use used using need want like way thing things something anything
        """.split()
    )

    def _unknown_key_terms(self, text: str) -> List[str]:
        """Content words the language model has never seen, longest first."""
        vocab = getattr(self.agent.lm, "vocab", None)
        if not vocab:
            return []
        known = {str(w).lower() for w in vocab}
        out: List[str] = []
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", text):
            word = raw.lower().strip("'-")
            if len(word) < 3 or word in self._STOP_TERMS or word in known:
                continue
            out.append(raw)
        out.sort(key=len, reverse=True)
        return out
