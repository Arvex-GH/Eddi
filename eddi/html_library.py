"""A written HTML component library for Eddi.

Eddi cannot *generate* good-looking HTML from its n-gram model - the model has
no notion of tag nesting or CSS.  What it can do is **compose** a page from
primitives that were written into memory, which is the whole point of Eddi's
architecture: quality comes from what is written, not from what is sampled.

The library is layered:

``DESIGN_TOKENS``  a small design system (colours, spacing, radii, fonts)
``CSS``            one stylesheet built from those tokens
``COMPONENTS``     composable, self-contained HTML snippets
``LAYOUTS``        page-level skeletons (centered, hero, dashboard, article)
``spec_from_request`` turn "make a landing page for X" into a written spec

Everything is plain strings with ``{slot}`` substitution, so a component can be
taught, inspected, overridden or extended - it lives in the knowledge base, not
in hidden model weights.
"""

from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------- tokens
DESIGN_TOKENS: Dict[str, str] = {
    # palette: neutral scale + one accent, tuned for contrast on light bg
    "bg": "#f7f8fa",
    "surface": "#ffffff",
    "ink": "#12151a",
    "muted": "#5c6472",
    "line": "#e3e6ec",
    "accent": "#3b5bfd",
    "accent_ink": "#ffffff",
    "accent_soft": "#eef1ff",
    # spacing / shape / type
    "space": "8px",
    "radius": "14px",
    "radius_sm": "9px",
    "shadow": "0 1px 2px rgba(16,24,40,.06), 0 8px 24px rgba(16,24,40,.08)",
    "font": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
}

CSS = """
:root {{
  --bg: {bg}; --surface: {surface}; --ink: {ink}; --muted: {muted};
  --line: {line}; --accent: {accent}; --accent-ink: {accent_ink};
  --accent-soft: {accent_soft}; --space: {space}; --radius: {radius};
  --radius-sm: {radius_sm}; --shadow: {shadow};
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  background: var(--bg); color: var(--ink);
  font-family: {font};
  font-size: 16px; line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 48px 24px; }}
.narrow {{ max-width: 720px; }}
h1, h2, h3 {{ line-height: 1.2; letter-spacing: -.02em; margin: 0 0 .5em; }}
h1 {{ font-size: clamp(2rem, 5vw, 3rem); font-weight: 800; }}
h2 {{ font-size: 1.5rem; font-weight: 700; }}
h3 {{ font-size: 1.1rem; font-weight: 650; }}
p {{ margin: 0 0 1em; color: var(--muted); }}
.lead {{ font-size: 1.15rem; color: var(--muted); max-width: 60ch; }}
a {{ color: var(--accent); }}
.btn {{
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--accent); color: var(--accent-ink);
  border: 0; border-radius: var(--radius-sm);
  padding: 12px 20px; font: inherit; font-weight: 650;
  cursor: pointer; text-decoration: none;
  transition: transform .12s ease, box-shadow .12s ease, background .12s ease;
}}
.btn:hover {{ transform: translateY(-1px); box-shadow: var(--shadow); }}
.btn:active {{ transform: translateY(0); }}
.btn.secondary {{
  background: var(--surface); color: var(--ink);
  border: 1px solid var(--line); box-shadow: none;
}}
.card {{
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 24px; box-shadow: var(--shadow);
}}
.grid {{ display: grid; gap: 24px; }}
.grid.three {{ grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
.hero {{ text-align: center; padding: 72px 24px 40px; }}
.hero .lead {{ margin: 0 auto 2em; }}
.pill {{
  display: inline-block; background: var(--accent-soft); color: var(--accent);
  border-radius: 999px; padding: 6px 14px; font-size: .85rem; font-weight: 650;
}}
nav {{ display: flex; align-items: center; justify-content: space-between;
       gap: 16px; padding: 18px 24px; border-bottom: 1px solid var(--line);
       background: var(--surface); }}
nav .brand {{ font-weight: 800; letter-spacing: -.02em; }}
nav .links {{ display: flex; gap: 20px; font-size: .95rem; }}
nav .links a {{ color: var(--muted); text-decoration: none; }}
nav .links a:hover {{ color: var(--ink); }}
table {{ width: 100%; border-collapse: collapse; background: var(--surface);
         border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }}
th, td {{ text-align: left; padding: 12px 16px; border-bottom: 1px solid var(--line); }}
th {{ font-size: .85rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }}
tr:last-child td {{ border-bottom: 0; }}
footer {{ border-top: 1px solid var(--line); color: var(--muted);
          font-size: .9rem; padding: 24px; text-align: center; }}
code {{ background: var(--accent-soft); color: var(--accent);
        padding: 2px 6px; border-radius: 6px; font-size: .9em; }}
@media (max-width: 640px) {{
  .wrap {{ padding: 28px 16px; }}
  .hero {{ padding: 48px 16px 24px; }}
  nav .links {{ display: none; }}
}}
"""

# ---------------------------------------------------------------- components
COMPONENTS: Dict[str, str] = {
    "page": (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>{title}</title>\n"
        "<style>\n{css}\n</style>\n</head>\n"
        "<body>\n{body}\n</body>\n</html>"
    ),
    "nav": (
        '<nav><span class="brand">{brand}</span>'
        '<span class="links">{links}</span></nav>'
    ),
    "hero": (
        '<header class="hero">'
        '<span class="pill">{eyebrow}</span>'
        "<h1>{title}</h1>"
        '<p class="lead">{subtitle}</p>'
        "{actions}"
        "</header>"
    ),
    "actions": '<div>{buttons}</div>',
    "button": '<a class="btn" href="{href}">{label}</a>',
    "button_secondary": '<a class="btn secondary" href="{href}">{label}</a>',
    "button_js": '<button class="btn" onclick="{action}">{label}</button>',
    "card": '<section class="card"><h3>{title}</h3><p>{text}</p></section>',
    "grid_three": '<div class="grid three">{cards}</div>',
    "section": '<section class="wrap narrow"><h2>{title}</h2>{body}</section>',
    "table": "<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>",
    "footer": '<footer>{text}</footer>',
    "list": "<ul>{items}</ul>",
}

# ------------------------------------------------------------------- layouts
LAYOUTS: Dict[str, List[str]] = {
    # which components a layout uses, in order
    "landing": ["nav", "hero", "grid_three", "footer"],
    "article": ["nav", "section", "footer"],
    "dashboard": ["nav", "hero", "grid_three", "table", "footer"],
    "blank": ["nav", "hero", "footer"],
}


@dataclass
class CardSpec:
    title: str
    text: str


@dataclass
class PageSpec:
    """Everything needed to assemble a page, derived from a written request."""

    title: str
    brand: str
    eyebrow: str = "New"
    heading: str = ""
    subtitle: str = ""
    primary_label: str = "Get started"
    primary_href: str = "#"
    primary_js: Optional[str] = None
    secondary_label: str = ""
    secondary_href: str = "#"
    cards: List[CardSpec] = field(default_factory=list)
    nav_links: Sequence[str] = ()
    footer: str = ""
    layout: str = "landing"
    table_rows: Sequence[Sequence[str]] = ()
    table_head: Sequence[str] = (
        "Item", "Status", "Value",
    )


def _esc(text: str) -> str:
    return _html.escape(str(text), quote=True)


def _js_string(text: str) -> str:
    """Escape a message for embedding inside a single-quoted JS string."""
    return str(text).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")


# ------------------------------------------------------------ spec inference
_STOP = {
    "make", "create", "build", "generate", "write", "a", "an", "the", "me",
    "please", "for", "about", "with", "of", "page", "website", "site", "html",
    "simple", "nice", "good", "looking", "pretty", "beautiful", "basic",
    "landing", "dashboard", "article", "blog", "post", "admin", "blank", "and",
}


# Written content packs: topic keyword -> headline/eyebrow/cards.
# These are primitives too - add a pack and Eddi can build that kind of site.
CONTENT_PACKS: Dict[str, Dict[str, object]] = {
    "coffee": {
        "eyebrow": "Now open",
        "subtitle": "Small-batch roasts, pulled fresh every morning. Come for the espresso, stay for the light.",
        "primary": "See the menu",
        "secondary": "Find us",
        "cards": [
            ("Single origin", "Rotating beans from small farms, roasted in-house every week."),
            ("Brewed to order", "Espresso, filter and cold brew - nothing sits under a lamp."),
            ("Open early", "Doors at 7am, seven days a week. Free wifi, no rush."),
        ],
    },
    "server": {
        "eyebrow": "Live",
        "subtitle": "Current health of the fleet: requests, errors and latency at a glance.",
        "primary": "Refresh",
        "secondary": "Docs",
        "cards": [
            ("Uptime", "99.98% over the last 30 days across all regions."),
            ("Throughput", "1,284 requests per second at the current peak."),
            ("p95 latency", "184 ms - within the 250 ms service objective."),
        ],
    },
    "dog": {
        "eyebrow": "Good boy",
        "subtitle": "Everything you need to know about the world's most important dog.",
        "primary": "See photos",
        "secondary": "Treats",
        "cards": [
            ("Walks", "Three a day, rain or shine, preferably through the park."),
            ("Tricks", "Sit, stay, roll over, and a very convincing innocent face."),
            ("Naps", "Roughly 14 hours daily, always on the best cushion."),
        ],
    },
    "hdc": {
        "eyebrow": "Architecture",
        "subtitle": "Hyperdimensional computing binds concepts with XOR and bundles them with a majority vote.",
        "primary": "Read the paper",
        "secondary": "View code",
        "cards": [
            ("Writable", "Facts are written as hypervectors - no backpropagation required."),
            ("Composable", "Bind, unbind and bundle compose into arbitrary structures."),
            ("Robust", "Majority-vote bundles recover facts from noisy probes."),
        ],
    },
}


def _pick_content_pack(request: str, topic: str) -> Optional[Dict[str, object]]:
    haystack = f"{request} {topic}".lower()
    if any(word in haystack for word in ("coffee", "cafe", "espresso", "roastery")):
        return CONTENT_PACKS["coffee"]
    if any(word in haystack for word in ("server", "metric", "monitor", "infra", "dashboard", "latency")):
        return CONTENT_PACKS["server"]
    if any(word in haystack for word in ("dog", "puppy", "pet")):
        return CONTENT_PACKS["dog"]
    if any(word in haystack for word in ("hdc", "hyperdimensional", "vector symbolic")):
        return CONTENT_PACKS["hdc"]
    return None


def spec_from_request(request: str) -> PageSpec:
    """Turn a short natural request into a written :class:`PageSpec`.

    ``"make a landing page for a coffee shop"`` -> title/brand ``Coffee Shop``.
    """
    cleaned = re.sub(r"[^\w\s\-&']+", " ", request or "").strip()
    words = [w for w in cleaned.split() if w]
    topic_words = [w for w in words if w.lower() not in _STOP]
    topic = " ".join(topic_words).strip() or "Eddi"
    title = topic.title()

    lowered = request.lower()
    if "dashboard" in lowered or "admin" in lowered or "table" in lowered:
        layout = "dashboard"
    elif "article" in lowered or "blog" in lowered or "post" in lowered:
        layout = "article"
    elif "blank" in lowered:
        layout = "blank"
    else:
        layout = "landing"

    pack = _pick_content_pack(request, topic)
    if pack:
        spec = PageSpec(
            title=title,
            brand=title,
            eyebrow=str(pack["eyebrow"]),
            heading=title,
            subtitle=str(pack["subtitle"]),
            primary_label=str(pack["primary"]),
            secondary_label=str(pack["secondary"]),
            footer=f"\u00a9 {title}. Built with Eddi.",
            layout=layout,
            cards=[CardSpec(c[0], c[1]) for c in pack["cards"]],
        )
    else:
        spec = PageSpec(
            title=title,
            brand=title,
            heading=title,
            subtitle=f"Everything about {title.lower()}, in one place.",
            primary_label="Get started",
            secondary_label="Learn more",
            footer=f"\u00a9 {title}. Built with Eddi.",
            layout=layout,
            cards=[
                CardSpec("Overview", f"What {title.lower()} is and why it matters."),
                CardSpec("Details", "The specifics, written down so they can be checked."),
                CardSpec("Next steps", "Where to go from here."),
            ],
        )
    if layout == "dashboard":
        spec.table_rows = (
            ("Requests", "Healthy", "1,284"),
            ("Errors", "Watch", "12"),
            ("Latency p95", "Healthy", "184 ms"),
        )
    return spec


# --------------------------------------------------------------- assembly
class HtmlBuilder:
    """Assembles a page from components. Components are writable at runtime."""

    def __init__(self, components: Optional[Dict[str, str]] = None):
        self.components: Dict[str, str] = dict(components or COMPONENTS)
        self.css: str = CSS.format(**DESIGN_TOKENS)

    def teach_component(self, name: str, template: str) -> None:
        """Write a new component (or override one) without retraining."""
        self.components[name] = template

    def _render(self, name: str, **slots) -> str:
        template = self.components.get(name)
        if template is None:
            raise KeyError(f"no component named {name!r}")
        return template.format(**slots)

    def build(self, spec: PageSpec) -> str:
        layout = LAYOUTS.get(spec.layout, LAYOUTS["landing"])
        parts: List[str] = []
        for component in layout:
            if component == "nav":
                links = "".join(
                    f'<a href="#">{_esc(link)}</a>' for link in spec.nav_links
                ) or "<a href='#'>Home</a><a href='#'>Docs</a><a href='#'>About</a>"
                parts.append(self._render("nav", brand=_esc(spec.brand), links=links))
            elif component == "hero":
                if spec.layout == "dashboard":
                    parts.append(
                        f'<header class="hero" style="padding:40px 24px 16px;text-align:left">'
                        f'<h1 style="font-size:clamp(1.6rem,3vw,2.2rem)">'
                        f'{_esc(spec.heading or spec.title)}</h1>'
                        f'<p class="lead">{_esc(spec.subtitle)}</p></header>'
                    )
                    continue
                primary = (
                    self._render("button_js", href="#", label=_esc(spec.primary_label),
                                 action=_js_string(spec.primary_js) + ";return false")
                    if spec.primary_js
                    else self._render("button", href=_esc(spec.primary_href),
                                      label=_esc(spec.primary_label))
                )
                secondary = (
                    self._render("button_secondary", href=_esc(spec.secondary_href),
                                 label=_esc(spec.secondary_label))
                    if spec.secondary_label
                    else ""
                )
                parts.append(self._render(
                    "hero",
                    eyebrow=_esc(spec.eyebrow),
                    title=_esc(spec.heading or spec.title),
                    subtitle=_esc(spec.subtitle),
                    actions=self._render("actions", buttons=primary + secondary),
                ))
            elif component == "grid_three" and spec.cards:
                cards = "".join(
                    self._render("card", title=_esc(card.title), text=_esc(card.text))
                    for card in spec.cards
                )
                parts.append(f'<main class="wrap">{self._render("grid_three", cards=cards)}</main>')
            elif component == "section":
                parts.append(self._render(
                    "section", title=_esc(spec.heading or spec.title),
                    body=f"<p>{_esc(spec.subtitle)}</p>",
                ))
            elif component == "table" and spec.table_rows:
                head = "".join(f"<th>{_esc(h)}</th>" for h in spec.table_head)
                rows = "".join(
                    "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>"
                    for row in spec.table_rows
                )
                parts.append(
                    f'<main class="wrap">'
                    f'{self._render("table", head=head, rows=rows)}</main>'
                )
            elif component == "footer":
                parts.append(self._render("footer", text=_esc(spec.footer)))
        return self._render(
            "page", title=_esc(spec.title), css=self.css, body="\n".join(parts)
        )


def describe_library() -> str:
    """Human-readable list of what Eddi has written - useful for ``--ask``."""
    lines = [f"HTML components ({len(COMPONENTS)}):"]
    for name in sorted(COMPONENTS):
        lines.append(f"  {name}")
    lines.append(f"CSS tokens ({len(DESIGN_TOKENS)}): " + ", ".join(sorted(DESIGN_TOKENS)))
    lines.append("Layouts: " + ", ".join(sorted(LAYOUTS)))
    return "\n".join(lines)
