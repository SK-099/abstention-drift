"""Detect 'I don't know' style responses in generated text."""
import re

# Same wording as MTRAG's reference answers for unanswerable turns.
IDK_RESPONSE = "I'm sorry, but I don't have the answer to your question."

_PATTERNS = [
    r"\bi (do not|don't) (know|have (the|an|enough) (answer|information))",
    r"\bi('m| am) (sorry|afraid),? (but )?(i )?(do not|don't|cannot|can't|couldn't)",
    r"\b(cannot|can't|couldn't|unable to) (find|answer|determine|locate)",
    r"\b(no|not enough|insufficient) (relevant )?information (is )?(available|provided|in the (passages|documents|context))",
    r"\bthe (passages|documents|context) (do|does) not (contain|mention|provide|say)",
    r"\bnot (mentioned|specified|covered|stated) in the (passages|documents|context)",
]
_RX = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def is_idk(text: str) -> bool:
    """True if the response is (mainly) a refusal. Only the first 200 chars are checked,
    so a real answer that merely *mentions* missing detail later is not counted."""
    return bool(_RX.search((text or "")[:200]))
