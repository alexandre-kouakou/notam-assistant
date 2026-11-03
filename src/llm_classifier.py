"""
LLM zero-shot classification for NOTAMs using the Anthropic Claude API.

Sends each NOTAM to Claude with the 13 class names as the only classification
context (no definitions, no examples — that's run 005's variant). Parses the
response back to a class id.

Responses are cached to JSONL on disk, keyed by input text, so a failed or
interrupted run does not lose completed work and can be resumed.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.config import CLASS_NAMES

load_dotenv()  # load ANTHROPIC_API_KEY from .env

SYSTEM_PROMPT = (
    "You are classifying aviation NOTAMs (Notices to Airmen) into one of 13 "
    "categories."
)

# Infer definitions for the DEEL-AI classes. Replace any of these with the
# exact wording from the DEEL-AI dataset card if you have it — the LLM
# responds differently to precise definitions than to approximations. Inferred
# here so the experiment runs without requiring a manual fetch step.
CLASS_DEFINITIONS = {
    "Airspaces": "Airspace restrictions, temporary flight restrictions, special use airspace, FIR/UIR notices.",
    "Communication_and_Radar": "Air traffic control communications, radar services, ATC frequencies, surveillance systems.",
    "GPS": "GPS/GNSS signal outages, jamming, RAIM, SBAS availability.",
    "Landing_Navaids": "Approach and landing aids: ILS (localizer, glideslope), DME, marker beacons, approach lighting associated with an ILS.",
    "Obstacles": "Physical obstructions: towers, cranes, trees, terrain, construction equipment.",
    "Parking": "Aircraft parking, aprons, stands, apron/taxiway conversions to parking areas.",
    "Procedure": "Published instrument approach, departure, or arrival procedures; operational instructions about how a procedure must or must not be flown. Distinct from a navaid outage even when the procedure references a navaid.",
    "Runway": "Runway status, closures, surface conditions, runway markings.",
    "Services_and_Facilities": "ARFF, fuel, ground services, airport facilities.",
    "Signs_and_Lights": "Airfield lighting, signs, approach lighting systems (ALS), aerodrome beacons, light outages.",
    "Taxiway": "Taxiway status, closures, surface conditions.",
    "Terminal_or_Enroute_Navaids": "VOR, NDB, TACAN, waypoints — navaids not tied to an approach procedure.",
    "Wildlife": "Bird or animal hazards, wildlife strikes, wildlife activity near airfields.",
}


def build_prompt_few_shot(text: str, examples: dict[str, list[str]]) -> str:
    """
    Few-shot prompt: each class followed by its definition and two training
    examples drawn from the train split (not val — no leakage).
    """
    blocks = []
    for name in CLASS_NAMES:
        definition = CLASS_DEFINITIONS[name]
        ex_lines = "\n".join(f'  Example: "{e}"' for e in examples.get(name, []))
        blocks.append(f"- {name}: {definition}\n{ex_lines}")
    categories = "\n".join(blocks)
    return (
        f"Categories (each with a definition and two examples):\n{categories}\n\n"
        f'NOTAM text: "{text}"\n\n'
        "Respond with only the category name, exactly as written above. "
        "Do not include explanation, quotes, or punctuation."
    )


DEFAULT_MODEL = "claude-haiku-4-5"

def build_prompt(text: str) -> str:
    categories = "\n".join(f"- {name}" for name in CLASS_NAMES)
    return (
        f"Categories:\n{categories}\n\n"
        f'NOTAM text: "{text}"\n\n'
        "Respond with only the category name, exactly as written above. "
        "Do not include explanation, quotes, or punctuation."
    )


def parse_response(response: str) -> int:
    """
    Map a free-text response to a class id. Returns -1 if unparseable.

    Handles three common LLM behaviors: exact match, case variance, and
    writing "Landing Navaids" instead of "Landing_Navaids" (space instead of
    underscore). Falls back to longest-substring match so a response like
    "The category is Landing_Navaids" still parses.
    """
    if not response or response.startswith("__ERROR__"):
        return -1

    cleaned = response.strip().strip("\"'`.,;:*").strip()
    candidates = [cleaned, cleaned.replace(" ", "_")]

    for c in candidates:
        if c in CLASS_NAMES:
            return CLASS_NAMES.index(c)

    lower_names = [n.lower() for n in CLASS_NAMES]
    for c in candidates:
        if c.lower() in lower_names:
            return lower_names.index(c.lower())

    normalized = cleaned.lower().replace(" ", "_")
    for i, name in sorted(enumerate(CLASS_NAMES), key=lambda x: -len(x[1])):
        if name.lower() in normalized:
            return i

    return -1


class LLMClassifier:
    """
    Zero-shot Claude classifier with disk-cached responses and threaded calls.

    Caching: each (text -> response) pair is appended to a JSONL file as it
    completes. Re-running the same configuration skips cached texts, so a
    crash at example 400 does not cost the first 400 API calls.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_path: Path | None = None,
        n_workers: int = 8,
        prompt_variant: str = "zero_shot",
        few_shot_examples: dict[str, list[str]] | None = None,
    ):
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        self.n_workers = n_workers
        self.prompt_variant = prompt_variant
        self.few_shot_examples = few_shot_examples or {}
        self._client = anthropic.Anthropic()
        self._cache = self._load_cache()

    def _build_prompt(self, text: str) -> str:
        if self.prompt_variant == "few_shot":
            return build_prompt_few_shot(text, self.few_shot_examples)
        return build_prompt(text)  # existing zero-shot builder

    def _call(self, text: str) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=128,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._build_prompt(text)}],
        )
        for block in message.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise RuntimeError(f"No text block in response: {message.content}")

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path and self.cache_path.exists():
            cache = {}
            with open(self.cache_path) as f:
                for line in f:
                    entry = json.loads(line)
                    cache[entry["text"]] = entry["response"]
            return cache
        return {}

    def _save_cache_entry(self, text: str, response: str) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"text": text, "response": response}) + "\n")


    def _classify_one(self, text: str) -> str:
        if text in self._cache:
            return self._cache[text]
        last_error = None
        for attempt in range(3):
            try:
                response = self._call(text)
                self._save_cache_entry(text, response)
                return response
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                time.sleep(1 + attempt)
        # Cache the failure too, so the file exists and is inspectable
        self._save_cache_entry(text, f"__ERROR__: {last_error}")
        return f"__ERROR__: {last_error}"

    def predict(self, texts: list[str]) -> list[int]:
        responses: list[str | None] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {
                pool.submit(self._classify_one, t): i
                for i, t in enumerate(texts)
            }
            for done, fut in enumerate(as_completed(futures), start=1):
                idx = futures[fut]
                responses[idx] = fut.result()
                if done % 100 == 0:
                    print(f"  {done}/{len(texts)} classified")
        return [parse_response(r or "") for r in responses]