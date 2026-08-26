import re
import random

# Existing English greetings are preserved.
_ENGLISH_TRIGGERS = {
    "hi", "hello", "hey", "goodmorning", "goodafternoon", "goodevening",
    "good day", "howdy", "greetings", "hi there", "hello there", "hey there",
    "what's up", "sup", "yo",
    "good morning", "good afternoon", "good evening",
}

_CEBUANO_TRIGGERS = {
    "maayong buntag",
    "maayong udto",
    "maayong hapon",
    "maayong gabii",
    "maayong adlaw",
    "kumusta ka",
    "kamusta ka",
    "kumusta",
    "kamusta",
    "musta",
    "salamat",
    "daghang salamat",
    "salamat kaayo",
}

_FILIPINO_TRIGGERS = {
    "magandang umaga",
    "magandang tanghali",
    "magandang hapon",
    "magandang gabi",
    "magandang araw",
    "kamusta",
    "musta",
    "salamat",
    "maraming salamat",
}

_ENGLISH_RESPONSES = [
    "Hello! I'm **KALAW**, your CPSU Faculty Manual assistant.\n\nI can help you with policies, procedures, leave benefits, faculty ranks, and anything covered in the Faculty Manual. What would you like to know?",
    "Hi there! Welcome — I'm **KALAW**, the CPSU Faculty Manual chatbot.\n\nFeel free to ask me about faculty policies, duties, benefits, or any section of the manual. How can I assist you today?",
    "Good day! I'm **KALAW**, here to help you navigate the CPSU Faculty Manual.\n\nAsk me anything about faculty rules, leave entitlements, promotions, or academic procedures. What's your question?",
    "Hey! I'm **KALAW** — your go-to guide for the CPSU Faculty Manual.\n\nWhether it's about teaching loads, leave policies, or faculty obligations, I'm ready to help. What do you need?"
]

_CEBUANO_RESPONSES = {
    "maayong buntag": "Maayong buntag! Ako si **KALAW**, ang imong CPSU Faculty Manual assistant.\n\nUnsa akong ikatabang nimo bahin sa Faculty Manual?",
    "maayong udto": "Maayong udto! Ako si **KALAW**, ang imong CPSU Faculty Manual assistant.\n\nUnsa akong ikatabang nimo bahin sa Faculty Manual?",
    "maayong hapon": "Maayong hapon! Ako si **KALAW**, ang imong CPSU Faculty Manual assistant.\n\nUnsa akong ikatabang nimo bahin sa Faculty Manual?",
    "maayong gabii": "Maayong gabii! Ako si **KALAW**, ang imong CPSU Faculty Manual assistant.\n\nUnsa akong ikatabang nimo bahin sa Faculty Manual?",
    "maayong adlaw": "Maayong adlaw! Ako si **KALAW**, ang imong CPSU Faculty Manual assistant.\n\nUnsa akong ikatabang nimo?",
    "kumusta ka": "Maayo ra, salamat! Ako si **KALAW**. Unsa akong ikatabang nimo bahin sa CPSU Faculty Manual?",
    "kamusta ka": "Maayo ra, salamat! Ako si **KALAW**. Unsa akong ikatabang nimo bahin sa CPSU Faculty Manual?",
    "kumusta": "Maayong adlaw! Kumusta? Unsa akong ikatabang nimo bahin sa CPSU Faculty Manual?",
    "kamusta": "Maayong adlaw! Kumusta? Unsa akong ikatabang nimo bahin sa CPSU Faculty Manual?",
    "musta": "Maayong adlaw! Kumusta? Unsa akong ikatabang nimo bahin sa CPSU Faculty Manual?",
    "salamat": "Walay sapayan! Kung naa kay pangutana bahin sa CPSU Faculty Manual, pangutana lang.",
    "daghang salamat": "Walay sapayan! Kung naa pa kay pangutana bahin sa CPSU Faculty Manual, pangutana lang.",
    "salamat kaayo": "Walay sapayan! Nalipay ko nga makatabang."
}

_FILIPINO_RESPONSES = {
    "magandang umaga": "Magandang umaga! Ako si **KALAW**, ang iyong CPSU Faculty Manual assistant.\n\nAno ang maitutulong ko sa iyo tungkol sa Faculty Manual?",
    "magandang tanghali": "Magandang tanghali! Ako si **KALAW**, ang iyong CPSU Faculty Manual assistant.\n\nAno ang maitutulong ko sa iyo tungkol sa Faculty Manual?",
    "magandang hapon": "Magandang hapon! Ako si **KALAW**, ang iyong CPSU Faculty Manual assistant.\n\nAno ang maitutulong ko sa iyo tungkol sa Faculty Manual?",
    "magandang gabi": "Magandang gabi! Ako si **KALAW**, ang iyong CPSU Faculty Manual assistant.\n\nAno ang maitutulong ko sa iyo tungkol sa Faculty Manual?",
    "magandang araw": "Magandang araw! Ako si **KALAW**, ang iyong CPSU Faculty Manual assistant.\n\nAno ang maitutulong ko sa iyo?",
    "kamusta": "Magandang araw! Kumusta? Ano ang maitutulong ko tungkol sa CPSU Faculty Manual?",
    "musta": "Magandang araw! Kumusta? Ano ang maitutulong ko tungkol sa CPSU Faculty Manual?",
    "salamat": "Walang anuman! Kung may tanong ka tungkol sa CPSU Faculty Manual, magtanong ka lang.",
    "maraming salamat": "Walang anuman! Kung may iba ka pang tanong tungkol sa CPSU Faculty Manual, magtanong ka lang.",
}


def _normalize(query: str) -> str:
    q = re.sub(r"[^a-z0-9\s']", " ", query.lower()).strip()
    return re.sub(r"\s+", " ", q)


def greeting_match(query: str) -> str | None:
    """
    Return an instant response only for a simple greeting.

    Real faculty questions continue through the existing KALAW pipeline.
    No retrieval, embedding, ranking, or LLM logic is performed here.
    """
    q = _normalize(query)

    # Keep the original behavior: greetings can include the name KALAW.
    q_without_kalaw = re.sub(r"\bkalaw\b", "", q)
    q_without_kalaw = re.sub(r"\s+", " ", q_without_kalaw).strip()

    # Exact English greeting -> existing English responses.
    if q_without_kalaw in _ENGLISH_TRIGGERS:
        return random.choice(_ENGLISH_RESPONSES)

    # Cebuano is checked before Filipino for shared words such as "kumusta".
    if q_without_kalaw in _CEBUANO_TRIGGERS:
        return _CEBUANO_RESPONSES.get(q_without_kalaw)

    # Filipino-specific greetings.
    if q_without_kalaw in _FILIPINO_TRIGGERS:
        return _FILIPINO_RESPONSES.get(q_without_kalaw)

    # Preserve support for greetings followed by extra words such as
    # "hello kalaw" or "maayong buntag kalaw" while avoiding
    # interception of actual Faculty Manual questions.
    for trigger in sorted(_ENGLISH_TRIGGERS, key=len, reverse=True):
        if q_without_kalaw.startswith(trigger + " ") and len(q_without_kalaw.split()) <= len(trigger.split()) + 2:
            return random.choice(_ENGLISH_RESPONSES)

    for trigger in sorted(_CEBUANO_TRIGGERS, key=len, reverse=True):
        if q_without_kalaw.startswith(trigger + " ") and len(q_without_kalaw.split()) <= len(trigger.split()) + 2:
            return _CEBUANO_RESPONSES.get(trigger)

    for trigger in sorted(_FILIPINO_TRIGGERS, key=len, reverse=True):
        if q_without_kalaw.startswith(trigger + " ") and len(q_without_kalaw.split()) <= len(trigger.split()) + 2:
            return _FILIPINO_RESPONSES.get(trigger)

    return None