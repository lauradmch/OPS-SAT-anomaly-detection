"""Data-to-text NLG. Numbers are owned by deterministic code; the LLM writes number-free framing only."""
import re
import os
from typing import Protocol

_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _numeral_tokens(text: str) -> list[str]:
    """Every numeral token in a string, e.g. ['0.94', '60', '1.0']."""
    return _NUM.findall(text)

### Type-hint rule

def infer_type_hint(ae_pct: float, hp_pct: float, is_anomaly: bool, margin: float = 0.10) -> str:
    """Deterministic type hint -> grounded fact for LLM report"""
    if not is_anomaly:
        return "nominal"
    if abs(ae_pct - hp_pct) <= margin:
        return "mixed"
    return "shape" if ae_pct > hp_pct else "transient"

### Grounded facts renderer: numbers become strings

def build_facts(pred: dict) -> dict[str, str]:
    """Converts the detector's numeric output into strings for ground-facts rendering"""
    facts = {
        "channel" : str(pred["channel"]),
        "verdict" : "ANOMALY" if pred["is_anomaly"] else "NOMINAL",
        "score" : f"{pred['score']:.2f}",
        "threshold" : f"{pred['threshold']:.2f}",
        "ae_pct" : f"{pred['ae_pct']:.2f}",
        "hp_pct" : f"{pred['hp_pct']:.2f}",
        "n_points" : str(int(pred["n_points"])),
        "model_version" : str(pred["model_version"]),
    }
    return facts


def allowed_tokens(facts: dict[str, str]) -> set[str]:
    allowed = set()
    for value in facts.values():
        allowed.update(_numeral_tokens(value))
    return allowed

### Faithfulness checker: extract numerals, exact-match against grounded tokens.
def faithfulness_check(text: str, facts: dict[str, str]) -> tuple[bool, list[str]]:
    allowed = allowed_tokens(facts)
    offending = [t for t in _numeral_tokens(text) if t not in allowed]
    return len(offending) == 0, offending


### Text-generation layer: Template + LLM 

TYPE_DESCRIPTIONS = {
    "nominal":   "no anomaly: both detector channels sit within their nominal reference bands",
    "shape":     "shape anomaly: the segment's morphology is hard to reconstruct, while its high-frequency content stays nominal",
    "transient": "high-frequency anomaly: unusual fast fluctuations dominate, while the overall shape reconstructs normally",
    "mixed":     "mixed anomaly: both the reconstruction and high-frequency channels are elevated together",
}


def _prompts(pred: dict, type_hint: str) -> tuple[str, str]:
    """Shared (system, user) prompt for any LLM narrator. Single source of truth."""
    system = (
        "You write concise anomaly summaries for spacecraft-telemetry operators. "
        "Hard rules, no exceptions:\n"
        "1. Write NO numbers, no digits and no number words "
        "(not 'ninety', 'two', 'first', etc.). All quantities are reported separately.\n"
        "2. Do not name, infer, or change the anomaly type; use only the type you are given.\n"
        "3. At most two sentences. Plain, factual, no speculation about causes."
    )
    user = (
        f"Verdict: {'anomaly detected' if pred['is_anomaly'] else 'nominal'}.\n"
        f"Anomaly type (fixed, describe this, do not change it): {type_hint} "
        f"-> {TYPE_DESCRIPTIONS[type_hint]}\n"
        f"Channel: {pred['channel']}\n"
        "Write the number-free operator summary now."
    )
    return system, user


class TemplateNarrator:
    name = "template"

    def narrate(self, pred: dict, type_hint: str) -> str:
        lead = ("The segment triggered the anomaly detector."
                if pred["is_anomaly"]
                else "The segment scored within nominal limits.")
        return f"{lead} {TYPE_DESCRIPTIONS[type_hint].capitalize()}."

class LLMNarrator:
    name = "llm"

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 200):
        import anthropic                       # lazy: optional dependency
        self._client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
        self._model = model
        self._max_tokens = max_tokens

    def narrate(self, pred: dict, type_hint: str) -> str:
        system, user = _prompts(pred, type_hint)
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content
                       if getattr(b, "type", "") == "text").strip()


class MistralNarrator:
    name = "mistral"                          # != "template", so fallback still triggers

    def __init__(self, model: str = "mistral-small-latest", max_tokens: int = 200):
        from mistralai import Mistral         # lazy: optional dependency
        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self._model = model
        self._max_tokens = max_tokens

    def narrate(self, pred: dict, type_hint: str) -> str:
        system, user = _prompts(pred, type_hint)
        resp = self._client.chat.complete(
            model=self._model,
            temperature=0,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},   # Mistral: system is a message
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()


### Composer: glue + fallback

def facts_block(facts: dict[str, str], type_hint: str) -> str:
    return (
        f"Channel {facts['channel']}: {facts['verdict']} "
        f"(score {facts['score']} vs threshold {facts['threshold']}). "
        f"Reconstruction percentile {facts['ae_pct']}, "
        f"high-pass percentile {facts['hp_pct']}, over {facts['n_points']} points. "
        f"Type: {type_hint}. Model {facts['model_version']}."
    )

def default_narrator() -> "Narrator":
    """Pick a narrator by available key: Mistral, then Anthropic, else template."""
    if os.getenv("MISTRAL_API_KEY"):
        try:
            return MistralNarrator()
        except Exception:
            pass            # missing SDK / bad client -> try next
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return LLMNarrator()
        except Exception:
            pass
    return TemplateNarrator()   # safe offline default

def render_report(pred: dict, narrator: "Narrator | None" = None, margin: float = 0.10) -> dict:
    narrator = narrator or default_narrator()
    facts = build_facts(pred)
    type_hint = infer_type_hint(pred["ae_pct"], pred["hp_pct"], pred["is_anomaly"], margin)
    block = facts_block(facts, type_hint)

    def compose(nar):
        narrative = nar.narrate(pred, type_hint).strip()
        return narrative, f"{narrative}\n\n{block}"

    used = narrator
    try:
        narrative, report = compose(narrator)
        faithful, offending = faithfulness_check(report, facts)
    except Exception:
        used = TemplateNarrator()          # API failure -> deterministic fallback
        narrative, report = compose(used)
        faithful, offending = faithfulness_check(report, facts)

    return {
        "channel": facts["channel"],
        "report": report,
        "type_hint": type_hint,
        "is_anomaly": pred["is_anomaly"],
        "faithful": faithful,
        "offending_numbers": offending,
        "generator": used.name,   
    }

### Endpoint: to FastAPI