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
import math
import operator
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .memory import identifiers_of, terms_of

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


def _query_score_ceiling(memory, query: str) -> float:
    """The strongest score this query could reach in this store.

    Computed by scoring the query against a synthetic chunk containing every
    query term - i.e. what a perfect match would look like. Normalising against
    it makes the acceptance threshold portable across corpus sizes.
    """
    terms = terms_of(query)
    idents = identifiers_of(query)
    if not terms and not idents:
        return 0.0
    n_docs = max(1, len(memory))
    total = 0.0
    for term in terms:
        df = len(memory._postings.get(term, []))
        if not df:
            continue
        total += math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
    # The phrase boost is part of the score the searcher can award, so it has to
    # be part of the ceiling too - otherwise a phrase hit scores above "perfect"
    # and the ratio stops meaning anything.
    if len(" ".join(query.lower().split())) > 8:
        total += 3.0
    return total + (6.0 * len(idents))


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


#: Objects that assert a property holds. A relation question may only be
#: answered from these, because any other object is a *value* ("Berlin"), not
#: a verdict, and "what is the capital of France" is not that kind of question.
_AFFIRMATIVE = frozenset({"yes", "true", "1"})


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
            self._handle_natural_math,
            self._handle_teach_rule,
            self._handle_teach_alias,
            self._handle_teach,
            self._handle_tell_about_self,
            self._handle_ask_about_self,
            self._handle_birth_year,
            self._handle_chain,
            self._handle_analogy,
            self._handle_relations,
            self._handle_false_premise,
            self._handle_who_verb,
            self._handle_when,
            self._handle_relation_question,
            self._handle_phrase_lookup,
            self._handle_types_of,
            self._handle_compare,
            self._handle_count,
            self._handle_reverse,
            self._handle_ask,
            self._handle_about,
            self._handle_greeting,
        ):
            answer = handler(text)
            if answer is not None:
                return answer
        # Retrieval from memory is the last resort before free-form generation:
        # structured knowledge wins, and stored text beats a model that will
        # always say *something* whether or not it knows.
        return self._handle_memory(text) or self._handle_generate(text)

    # -------------------------------------------------------------- memory
    # Retrieval over stored text. Deliberately placed *after* the structured
    # handlers, which are exact, and *before* the language model, which will
    # always produce something whether or not it knows.
    _MIN_MEMORY_SCORE = 1.2

    _QUESTION_MARKERS = re.compile(
        r"\b(?:what|which|who|whom|whose|when|where|why|how|is|are|was|were|"
        r"does|do|did|can|could|should|would|will)\b",
        re.IGNORECASE,
    )

    #: "tell me about water" asks for something just as much as "what is water".
    _REQUEST_MARKERS = re.compile(
        r"^\s*(?:tell\s+me|explain|describe|show\s+me|give\s+me|list|"
        r"what\s+do\s+you\s+know|do\s+you\s+know)\b",
        re.IGNORECASE,
    )

    def _looks_like_question(self, text: str) -> bool:
        """Distinguish a question from a statement.

        A leading greeting is peeled off first: "Hi! Tell me about water" is a
        request, not a statement about oneself.
        """
        stripped = self._GREETING_OPENERS.sub("", text.strip()).strip(" ,.!")
        if not stripped:
            return False
        if stripped.endswith("?"):
            return True
        if self._REQUEST_MARKERS.search(stripped):
            return True
        # The interrogative has to *open* the sentence. Matching it anywhere
        # marks every declarative with a copula as a question: "the sky is blue"
        # becomes a question, so it is filed as one and can never be recalled.
        return bool(self._QUESTION_MARKERS.match(stripped))

    #: Symbols that are notation, not words. Replacing their underscores with
    #: spaces turns a correct answer into noise ("O of n log n"), so they pass
    #: through untouched.
    _NOTATION = frozenset({"O_of_1", "O_of_log_n", "O_of_n", "O_of_n_log_n", "O_of_n_squared"})

    @classmethod
    def _render(cls, value: str) -> str:
        """Storage-safe text shown to a human.

        Facts are written with underscores so a value survives tokenisation as a
        single symbol ("19.3_g_per_cm3", "no_gpu"). Those underscores are an
        encoding detail and must not reach the answer: the user asked for a
        density, not for a symbol name.
        """
        if value in cls._NOTATION:
            return value
        return value.replace("_", " ")

    def _handle_memory(self, text: str) -> Optional[Answer]:
        """Answer from verbatim stored text when it matches well enough.

        A retrieval hit is evidence, not proof: the score threshold is what
        separates "this span is about the question" from "these two chunks share
        the word 'the'". Below the threshold we return None and let the caller
        fall through to an honest refusal.
        """
        memory = getattr(self.agent, "memory", None)
        if memory is None or not len(memory):
            return None
        if not self._looks_like_question(text):
            # A statement has just been stored by the caller; matching it against
            # itself and returning it as an answer is a tautology, not an answer.
            return Answer(
                "Noted - I've stored that.",
                kind="remembered",
                confidence=1.0,
                details=["written to verbatim memory"],
            )
        hits = memory.search(text, k=5)
        if not hits:
            return None

        # A stored question is weak evidence: a previously asked question
        # resembles the current one closely without containing any answer.
        def _is_question(chunk) -> bool:
            return chunk.source in ("question", "reply-question") or chunk.text.strip().endswith("?")

        # Never fall back to question chunks: if the only things that resemble
        # the query are earlier questions, memory has nothing to say and the
        # honest answer is to decline.
        pool = [h for h in hits if not _is_question(h.chunk)]
        if not pool:
            return None

        # Acceptance is decided by *coverage*, not by raw score. BM25 magnitudes
        # depend on corpus size, so any absolute threshold works on one store and
        # fails on another; and a ratio against the best possible score lets a
        # chunk through on a single common word. What actually distinguishes a
        # real match is how much of the question the chunk accounts for.
        q_terms = set(terms_of(text))
        q_idents = set(identifiers_of(text))
        wanted = max(1, len(q_terms))
        best = None
        best_cover = 0.0
        for hit in pool:
            matched = set(hit.matched_terms) & q_terms
            coverage = len(matched) / wanted
            if q_idents and hit.matched_identifiers:
                coverage = 1.0  # an exact identifier is decisive on its own
            # A single shared word is not a match. "hi" shares "hi" with
            # "hi, what is python" and would answer it with a greeting.
            if len(matched) < 2 and not (q_idents and hit.matched_identifiers):
                continue
            if coverage > best_cover:
                best, best_cover = hit, coverage
        if best is None or best_cover < 0.34:
            return None
        # A question that the fact base could answer is handled earlier; by the
        # time we are here the knowledge base had nothing, so pass the span on.
        quoted = best.text.strip()
        if len(quoted) > 420:
            quoted = quoted[:420].rsplit(" ", 1)[0] + "..."
        source = f" (from {best.chunk.source})" if best.chunk.source else ""
        return Answer(
            quoted + source,
            kind="memory",
            confidence=min(1.0, best.score / 8.0),
            details=[
                f"memory chunk #{best.chunk.id}, score {best.score:.2f}",
                "matched: " + ", ".join(
                    (best.matched_identifiers + best.matched_terms)[:6]
                ) if (best.matched_identifiers or best.matched_terms) else "phrase match",
            ],
        )

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
        lines = [f"{self._render(subject)}:"]
        for relation, obj, _conf in facts[:12]:
            lines.append(f"  {self._render(relation)}: {self._render(obj)}")
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

    #: "who invented the printing press" -> the fact is (printing_press,
    #: invented_by, Gutenberg). The verb reverses subject and object, so these
    #: need their own path; no subject-first handler can reach them.
    _WHO_VERB_Q = re.compile(
        # "the" is part of the captured name, not noise to discard: "the
        # Elements" is a book title and stripping the article turned it into
        # the bare symbol "element". resolve_entity copes with a stray article.
        r"^who\s+(invented|discovered|wrote|composed|painted|founded|created|"
        r"built|designed|developed|proposed|led|ruled)\s+(.+?)\s*\??$",
        re.IGNORECASE,
    )

    #: English past tense to the relation a fact would be stored under. The
    #: curriculum writes "invented_by", "discovered_by", "wrote" and so on, so
    #: the verb has to be translated, not merely reversed.
    _VERB_TO_RELATION = {
        "invented": ("invented_by", "inventor"),
        "discovered": ("discovered_by", "discoverer"),
        "wrote": ("wrote", "written_by", "author"),
        "composed": ("composed_by", "composer"),
        "painted": ("painted_by", "painter"),
        "founded": ("founded_by", "founder"),
        "created": ("created_by", "creator"),
        "built": ("built_by", "builder"),
        "designed": ("designed_by", "designer"),
        "developed": ("developed_by", "developer"),
        "proposed": ("proposed_by", "proposer"),
        "led": ("led_by", "leader"),
        "ruled": ("ruled", "ruler"),
    }

    def _handle_who_verb(self, text: str) -> Optional[Answer]:
        """Answer "who invented the printing press"."""
        match = self._WHO_VERB_Q.match(text.strip())
        if not match:
            return None
        verb = match.group(1).lower()
        thing_text = match.group(2).strip()
        thing = self.agent.resolve_entity(thing_text)
        if thing is None:
            return None
        candidates = self._VERB_TO_RELATION.get(verb, ())
        facts = self.agent.kb.facts
        for relation in candidates:
            values = [
                obj
                for subj, rel, obj, _hv in facts
                if subj == thing and rel == relation
            ]
            if not values:
                # The fact may be stored the other way round: "who wrote the
                # Elements" is (Euclid, wrote, the_Elements), where the person
                # is the subject.
                values = [
                    subj
                    for subj, rel, obj, _hv in facts
                    if rel == relation and obj == thing
                ]
            if values:
                values = list(dict.fromkeys(values))
                listing = " and ".join(self._render(v) for v in values)
                return Answer(
                    f"{listing}.",
                    kind="ask",
                    confidence=1.0,
                    details=[f"({thing}, {relation}) -> {listing}"],
                )
        return None

    #: "when was the printing press invented" -> (printing_press,
    #: invented_year, 1440). The year lives under a relation derived from the
    #: verb, so the question has to be mapped rather than looked up directly.
    _WHEN_Q = re.compile(
        r"^when\s+(?:was|were|did)\s+(.+?)\s+"
        r"(invented|discovered|built|founded|written|published|born|signed|"
        r"completed|created|established)\s*\??$",
        re.IGNORECASE,
    )

    _VERB_TO_YEAR_RELATION = {
        "invented": ("invented_year", "year"),
        "discovered": ("discovered_year", "year"),
        "built": ("built_year", "built_year_BCE", "year"),
        "founded": ("founded_year", "founded_year_BCE", "year"),
        "written": ("written_year", "year"),
        "published": ("published_year", "published_year_BCE", "year"),
        "signed": ("signed_year", "year"),
        "completed": ("completed_year", "year"),
        "created": ("created_year", "year"),
        "established": ("established_year", "year"),
        "born": ("born_year", "born_year_BCE", "year"),
    }

    def _handle_when(self, text: str) -> Optional[Answer]:
        """Answer "when was the printing press invented"."""
        match = self._WHEN_Q.match(text.strip())
        if not match:
            return None
        thing_text, verb = match.group(1).strip(), match.group(2).lower()
        thing = self.agent.resolve_entity(thing_text)
        if thing is None:
            return None
        for relation in self._VERB_TO_YEAR_RELATION.get(verb, ()):  # type: ignore[arg-type]
            values = [
                obj
                for subj, rel, obj, _hv in self.agent.kb.facts
                if subj == thing and rel == relation
            ]
            if values:
                values = list(dict.fromkeys(values))
                listing = " and ".join(self._render(v) for v in values)
                label = "BCE" if relation.endswith("BCE") else ""
                text_out = f"{listing} {label}".strip()
                return Answer(
                    f"{text_out}.",
                    kind="ask",
                    confidence=1.0,
                    details=[f"({thing}, {relation}) -> {listing}"],
                )
        return None

    def _handle_relation_question(self, text: str) -> Optional[Answer]:
        """Answer "what is the largest ocean" when the relation is the topic.

        "what is the largest ocean" names no subject. The knowledge is stored as
        ("Pacific", "largest_ocean", "yes"), so the question is really "which
        subject has this property true". Only facts whose object is an
        affirmation qualify -- otherwise "what is the capital of France" would be
        swallowed here as "the relation 'capital_of'" and answered with a city.
        """
        match = re.match(
            r"^(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(.+?)\s*\??$",
            text.strip(), re.IGNORECASE,
        )
        if not match:
            return None
        # "<relation> of <subject>" is a *lookup*, not a relation question. It
        # must reach _answer_attribute, which knows the subject. Claiming it
        # here answered "what is the atomic number of gold" with "hydrogen",
        # the subject of whichever fact happened to be scanned first.
        if re.search(r"\b(?:of|for)\b", match.group(1), re.IGNORECASE):
            return None
        relation = self.agent.resolve_relation(match.group(1))
        if relation is None:
            return None
        subjects = [
            subj
            for subj, rel, obj, _hv in self.agent.kb.facts
            if rel == relation and obj in _AFFIRMATIVE
        ]
        if not subjects:
            return None
        subjects = list(dict.fromkeys(subjects))
        listing = ", ".join(self._render(x) for x in subjects)
        return Answer(
            f"{listing}.",
            kind="ask",
            confidence=1.0,
            details=[f"subjects where ({relation}) holds: {listing}"],
        )

    def _handle_phrase_lookup(self, text: str) -> Optional[Answer]:
        """Answer 'what is the <relation> for <subject>' without mistaking a
        preposition for the relation.

        'what is the default port for HTTPS' matched the generic
        'what is <relation> of/for <subject>' path, took 'for' as the relation
        name, and replied "I know the relation 'form'". Words that exist only to
        join a question are not relations.
        """
        match = re.match(
            r"^what\s+is\s+(?:the\s+)?(.+?)\s+(?:for|of)\s+(?:the\s+)?(.+?)\s*\??$",
            text.strip(), re.IGNORECASE,
        )
        if not match:
            return None
        if match.group(1) is not None:
            relation_text, subject_text = match.group(1).strip(), match.group(2).strip()
        elif match.group(3) is not None:
            relation_text, subject_text = match.group(3).strip(), match.group(4).strip()
        else:
            relation_text, subject_text = match.group(5).strip(), match.group(6).strip()
        subject = self.agent.resolve_entity(subject_text)
        if subject is None:
            return None
        relation = self.agent.resolve_relation(relation_text)
        if relation is None:
            return None
        # Only claim the question if this subject actually has that relation.
        # Otherwise fall through, so 'what is the capital of France' still
        # reaches the handler that sweeps every capital.
        hits = self.agent.kb.query(subject, relation)
        if not hits:
            return None
        return Answer(
            f"{self._render(hits[0][0])}",
            kind="ask",
            confidence=1.0,
            details=[f"memory: ({subject}, {relation}) -> {hits[0][0]}"],
        )

    def _handle_types_of(self, text: str) -> Optional[Answer]:
        """Answer 'what orbits the Sun' - a class question, not an instance one.

        'which planet does the Moon orbit' asks for one entity; 'what orbits the
        Sun' asks for everything of that kind. The base had no way to sweep a
        relation backwards across every subject.
        """
        match = re.match(
            r"^what\s+(?:things\s+)?(.+?)\s+(?:the\s+)?(.+?)\s*\??$",
            text.strip(), re.IGNORECASE,
        )
        if not match:
            return None
        relation_text, object_text = match.group(1).strip(), match.group(2).strip()
        relation = self.agent.resolve_relation(relation_text)
        target = self.agent.resolve_entity(object_text)
        if relation is None or target is None:
            return None
        subjects = [
            s for s, r, o, _h in self.agent.kb.facts
            if r == relation and o == target
        ]
        if not subjects:
            return None
        subjects = list(dict.fromkeys(subjects))
        listing = ", ".join(s.replace("_", " ") for s in subjects)
        return Answer(
            f"{listing}.",
            kind="reverse",
            confidence=1.0,
            details=[f"all subjects with --{relation}--> {target}"],
        )

    # ------------------------------------------------------- reverse lookup
    def _handle_reverse(self, text: str) -> Optional[Answer]:
        """Answer 'what is Paris the capital of?' - the fact is stored backwards.

        The base writes capital_of as (Paris, capital_of, France). Asking what
        Paris *is the capital of* means walking the inverse direction, which no
        subject->object handler could do.
        """
        match = re.match(
            r"^(?:what|which)\s+(?:is|was)?\s*(.+?)\s+(?:the\s+)?(.+?)\s+of\s*\??$",
            text, re.IGNORECASE,
        )
        if not match:
            return None
        subject_text, relation_text = match.group(1).strip(), match.group(2).strip()
        subject = self.agent.resolve_entity(subject_text)
        relation = self.agent.resolve_relation(relation_text)

        # The stored direction decides the reading. "capital_of" is stored
        # (Paris, capital_of, France), so "what is Paris the capital of" asks
        # for the object and "what is France the capital of" asks for the
        # subject -- the question cannot tell you which without the data.
        if relation is None:
            return None
        if subject is None:
            return None
        facts = self.agent.kb.facts
        values = [
            obj
            for subj, rel, obj, _hv in facts
            if rel == relation and subj == subject
        ]
        inverted = [
            subj
            for subj, rel, obj, _hv in facts
            if rel == relation and obj == subject and subj != subject
        ]
        # Prefer the direction that has data; if both do, answer both readings.
        values = list(dict.fromkeys(values))
        if not values and inverted:
            values = list(dict.fromkeys(inverted))
        if not values:
            return None
        pretty = " and ".join(self._render(v) for v in values)
        return Answer(
            f"{pretty}.",
            kind="reverse",
            confidence=1.0,
            details=[f"{subject} --{relation}--> {pretty}"],
        )

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
            lines.append(f"  {self._render(relation)}: {self._render(obj)}")
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

    # ---------------------------------------------------- natural arithmetic
    # Problems stated in words rather than symbols. The expression evaluator
    # next door only accepts arithmetic characters, so these need translating
    # first: "15% of 200" is not an expression, it is a sentence.
    _PERCENT_Q = re.compile(
        r"^(?:what\s+is\s+|calc(?:ulate)?\s+)?(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+"
        r"(-?\d+(?:\.\d+)?)\s*\??$",
        re.IGNORECASE,
    )
    _POWER_Q = re.compile(
        r"^(?:what\s+is\s+)?(-?\d+(?:\.\d+)?)\s*(?:to\s+the\s+power\s+of|"
        r"\^|raised\s+to)\s*(-?\d+)\s*\??$",
        re.IGNORECASE,
    )
    _SQRT_Q = re.compile(
        r"^(?:what\s+is\s+)?(?:the\s+)?square\s+root\s+of\s+(-?\d+(?:\.\d+)?)\s*\??$",
        re.IGNORECASE,
    )

    # Length in millimetres, time in seconds. Two separate tables because they
    # are not commensurable - converting metres to seconds is not a unit error
    # to be silently "fixed", it is a question with no answer.
    _LENGTH_MM = {
        "millimeter": 1, "millimetre": 1, "mm": 1,
        "centimeter": 10, "centimetre": 10, "cm": 10,
        "meter": 1000, "metre": 1000, "m": 1000,
        "kilometer": 1000000, "kilometre": 1000000, "km": 1000000,
    }
    _TIME_S = {
        "second": 1, "sec": 1, "s": 1,
        "minute": 60, "min": 60,
        "hour": 3600, "hr": 3600, "h": 3600,
        "day": 86400,
        "week": 604800,
        "year": 31536000,
    }

    @classmethod
    def _unit_factor(cls, name: str):
        """(factor, table_id) for a unit, tolerating plurals ('kilometers')."""
        word = name.strip().lower().replace(" ", "")
        for table_id, table in (("length", cls._LENGTH_MM), ("time", cls._TIME_S)):
            if word in table:
                return table[word], table_id
            if word.endswith("s") and word[:-1] in table:
                return table[word[:-1]], table_id
        return None, None

    def _handle_natural_math(self, text: str) -> Optional[Answer]:
        """Worded arithmetic: percentages, powers, roots, unit conversion.

        These all failed before because the evaluator only parses symbols, so
        '15% of 200' and 'square root of 144' fell through to the knowledge
        handlers and were answered as lookups for the number's name.
        """
        match = self._PERCENT_Q.match(text.strip())
        if match:
            pct, whole = float(match.group(1)), float(match.group(2))
            return Answer(
                f"{_format_number(pct / 100.0 * whole)}",
                kind="math",
                confidence=1.0,
                details=[f"{pct}% of {whole} = {pct}/100 * {whole}"],
            )

        match = self._POWER_Q.match(text.strip())
        if match:
            base, exp = float(match.group(1)), int(match.group(2))
            if abs(exp) > 1000:
                return Answer(
                    "That exponent is too large to compute usefully.", kind="math_error"
                )
            try:
                value = base ** exp
            except OverflowError:
                return Answer("That number is too large to represent.", kind="math_error")
            return Answer(f"{_format_number(value)}", kind="math", confidence=1.0,
                          details=[f"{base} ** {exp}"])

        match = self._SQRT_Q.match(text.strip())
        if match:
            value = float(match.group(1))
            if value < 0:
                return Answer(
                    "The square root of a negative number is not a real number.",
                    kind="math_error",
                )
            root = value ** 0.5
            return Answer(f"{_format_number(root)}", kind="math", confidence=1.0,
                          details=[f"sqrt({value})"])

        # "how many meters in 5 kilometers" / "5 km in meters"
        match = re.match(
            r"^(?:how\s+many\s+)?(\w+?)\s+(?:are\s+)?in\s+"
            r"(?:a|an|one)\s+(\w+?)\s*\??$|"
            r"^(?:how\s+many\s+)?(\w+?)\s+(?:are\s+)?in\s+"
            r"(-?\d+(?:\.\d+)?)\s*(\w+?)\s*\??$|"
            r"^(?:convert\s+)?(-?\d+(?:\.\d+)?)\s*(\w+?)\s+(?:to|in)\s+(\w+?)\s*\??$",
            text.strip(), re.IGNORECASE,
        )
        if match:
            if match.group(1) and match.group(2):
                # "how many seconds in a day"
                dest, amount, src = match.group(1), 1.0, match.group(2)
            elif match.group(3):
                dest, amount, src = match.group(3), float(match.group(4)), match.group(5)
            else:
                amount, src, dest = float(match.group(6)), match.group(7), match.group(8)
            src_factor, src_kind = self._unit_factor(src)
            dest_factor, dest_kind = self._unit_factor(dest)
            if src_factor is None or dest_factor is None:
                return None
            if src_kind != dest_kind:
                return Answer(
                    f"I can't convert {src.lower()} to {dest.lower()} - "
                    "those measure different things.",
                    kind="convert_error",
                )
            result = amount * src_factor / dest_factor
            return Answer(
                f"{_format_number(result)} {dest.lower()}",
                kind="convert",
                confidence=1.0,
                details=[f"{amount} {src.lower()} converted to {dest.lower()}"],
            )
        return None

    _FALSE_PREMISE_Q = re.compile(
        r"^why\s+(?:is|are|was|were)\s+(?:the\s+)?(.+?)\s+(?:a|an|the)\s+(.+?)\s*\??$",
        re.IGNORECASE,
    )

    def _handle_false_premise(self, text: str) -> Optional[Answer]:
        """Catch a question built on something the base contradicts.

        'Why is the Sun a planet?' presumes the Sun is a planet. Answering with
        the Sun's facts silently accepts the premise; the honest reply names it.
        Only fires when both entities are known and the claim conflicts with a
        written 'type'.
        """
        match = self._FALSE_PREMISE_Q.match(text.strip())
        if not match:
            return None
        subject_text, claim_text = match.group(1).strip(), match.group(2).strip()
        subject = self.agent.resolve_entity(subject_text)
        claim = self.agent.resolve_entity(claim_text)
        if subject is None:
            return None
        actual = None
        for relation, obj, _conf in self.agent.kb.facts_about(subject):
            if relation == "type":
                actual = obj
        if actual is None:
            return None
        if claim is not None and (claim == actual or claim in actual):
            # The premise is true; let the normal handlers answer it.
            return None
        return Answer(
            f"I can't answer that as asked, because {subject_text} is not "
            f"a {claim_text} - I have its type as {actual.replace('_', ' ')}.",
            kind="false_premise",
            confidence=1.0,
            details=[f"contradicts stored ({subject}, type, {actual})"],
        )

    # ------------------------------------------------------- comparison & count
    _COMPARE_Q = re.compile(
        r"^(?:which|what)\s+is\s+(faster|slower|bigger|smaller|larger|older|younger|"
        r"higher|lower|heavier|lighter)\s*,?\s*(.+?)\s+or\s+(?:a\s+|an\s+|the\s+)?(.+?)\s*\??$",
        re.IGNORECASE,
    )
    _SPEED_RANK = {"fastest_memory": 4, "slower_than_cache": 3,
                   "slower_than_RAM": 2, "slowest": 1}
    _RANK_ATTRS = {
        "faster": ("speed_rank", True), "slower": ("speed_rank", False),
        "higher": ("speed_rank", True), "lower": ("speed_rank", False),
    }
    # Base forms for the "I have no data" message. String surgery on the
    # comparative produced "bigg", which is exactly the kind of sloppy output
    # that makes a system look careless even when the logic is right.
    _ADJ_BASE = {
        "faster": "speed", "slower": "speed",
        "bigger": "size", "smaller": "size", "larger": "size",
        "older": "age", "younger": "age",
        "higher": "height", "lower": "height",
        "heavier": "weight", "lighter": "weight",
    }

    def _handle_compare(self, text: str) -> Optional[Answer]:
        """'which is faster, RAM or a register?' - compare written properties.

        Both sides must carry a comparable attribute; otherwise the question is
        answered about whichever entity happened to resolve.
        """
        match = self._COMPARE_Q.match(text.strip())
        if not match:
            return None
        adjective = match.group(1).lower()
        left_text, right_text = match.group(2).strip(), match.group(3).strip()

        attribute, want_high = self._RANK_ATTRS.get(adjective, ("speed_rank", True))
        scored = []
        for label in (left_text, right_text):
            entity = self.agent.resolve_entity(label)
            if entity is None:
                return None
            value = None
            for rel, obj, _conf in self.agent.kb.facts_about(entity):
                if rel == attribute:
                    value = obj
                    break
            if value is None:
                return Answer(
                    f"I don't know the {self._ADJ_BASE.get(adjective, 'value')} of "
                    f"{label}, so I can't say which is {adjective}.",
                    kind="unknown",
                    details=[f"missing ({entity}, {attribute})"],
                )
            scored.append((label, entity, self._SPEED_RANK.get(value, 0), value))

        (l_label, l_entity, l_rank, l_value), (r_label, r_entity, r_rank, r_value) = scored
        if l_rank == r_rank:
            winner = None
        else:
            left_wins = (l_rank > r_rank) == want_high
            winner = (l_label, l_entity, l_rank, l_value) if left_wins else (r_label, r_entity, r_rank, r_value)
        if winner is None:
            return Answer(
                f"They rank the same on {attribute}, so neither is {adjective}.",
                kind="compare",
                confidence=1.0,
            )
        return Answer(
            f"{winner[0].capitalize()} is {adjective}.\n"
            f"  {l_label}: {attribute} = {l_value}\n"
            f"  {r_label}: {attribute} = {r_value}",
            kind="compare",
            confidence=1.0,
            details=[f"compared written {attribute} values"],
        )

    _COUNT_Q = re.compile(
        # "how many planets does the solar system have" / "how many planets are
        # in the solar system": the relation is a plural noun and the subject
        # follows it. All three forms put (relation, subject) in groups 1/2,
        # 3/4 or 5/6 so one body can serve them.
        r"^how\s+many\s+(\w+)\s+(?:does|do)\s+(?:the\s+)?(.+?)\s+have\s*\??$|"
        r"^how\s+many\s+(\w+)\s+(?:are|is)\s+(?:there\s+)?(?:in|on)\s+(?:the\s+)?(.+?)\s*\??$|"
        r"^how\s+many\s+(.+?)\s+do\s+you\s+know\s+about\s+(?:the\s+)?(.+?)\s*\??$|"
        r"^how\s+many\s+facts\s+do\s+you\s+know\s*\??$",
        re.IGNORECASE,
    )

    def _handle_count(self, text: str) -> Optional[Answer]:
        """Count what is actually stored, rather than guessing."""
        match = self._COUNT_Q.match(text.strip())
        if not match:
            return None
        # The subject sits in group 2, 4 or 6 depending on which alternation
        # matched; the bare "how many facts" form has none of them.
        if all(match.group(i) is None for i in (2, 4, 6)):
            return Answer(
                f"I have {len(self.agent.kb)} facts written.",
                kind="count",
                confidence=1.0,
                details=["counted len(kb)"],
            )
        if match.group(1) is not None:
            relation_text, subject_text = match.group(1).strip(), match.group(2).strip()
        elif match.group(3) is not None:
            relation_text, subject_text = match.group(3).strip(), match.group(4).strip()
        else:
            relation_text, subject_text = match.group(5).strip(), match.group(6).strip()
        subject = self.agent.resolve_entity(subject_text)
        if subject is None:
            return Answer(
                f"I don't know anything about '{subject_text}'.",
                kind="unknown",
            )
        relation = self.agent.resolve_relation(relation_text)
        all_facts = self.agent.kb.facts_about(subject)
        named = [o for r, o, _c in all_facts if relation and r == relation]
        # A counted relation stores the number as the object ("8"), so answer the
        # value rather than listing it as though it were a name.
        if relation and len(named) == 1 and named[0].replace("-", "").isdigit():
            return Answer(
                f"{named[0]}.",
                kind="count",
                confidence=1.0,
                details=[f"({subject}, {relation}) -> {named[0]}"],
            )
        if relation:
            return Answer(
                f"{len(named)}: " + (", ".join(o.replace("_", " ") for o in named) or "none"),
                kind="count",
                confidence=1.0,
                details=[f"counted ({subject}, {relation}) objects"],
            )
        return Answer(
            f"{len(all_facts)} facts about {subject.replace('_', ' ')}.",
            kind="count",
            confidence=1.0,
            details=[f"counted facts_about({subject})"],
        )

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
        # 'help' is a request for the command list, not a pleasantry. It must be
        # routed before the small-talk gate, which would otherwise reject it.
        bare = text.strip()
        # a lone '?' strips to '' , so test the raw form as well
        if bare == "?" or bare.lower().strip("!. ") in {"help", "commands", "what can you do"}:
            return Answer(
                "I'm Eddi. Things I can actually do:\n"
                "  tell me about <subject>        facts I hold about it\n"
                "  what is the <relation> of <subject>\n"
                "  what <relation> is <subject>   (also: which ..., who ...)\n"
                "  how old am I / which year was I born\n"
                "  I am 16 years old              (tell me about yourself)\n"
                "  teach <subject> | <relation> | <object>\n"
                "  rule <rel1> => <rel2>          writable inference\n"
                "  alias <phrase> = <canonical>\n"
                "  chain <start> | <rel1> | <rel2>\n"
                "  Earth is to Sun as Moon is to ?\n"
                "  what is 2 + 2 * 10\n"
                "  how many <relation> do you know about <subject>",
                kind="help",
            )
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
