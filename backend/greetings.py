import random
import re


_GREETING_TRIGGERS = {
    "english": {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
        "howdy",
        "greetings",
        "hi there",
        "hello there",
        "hey there",
        "what's up",
        "sup",
        "yo",
    },

    "cebuano": {
        "kamusta",
        "kumusta",
        "musta",
        "kumusta ka",
        "maayong buntag",
        "maayong udto",
        "maayong hapon",
        "maayong gabii",
        "maayong adlaw",
    },

    "filipino": {
        "kamusta",
        "kumusta",
        "musta",
        "magandang umaga",
        "magandang hapon",
        "magandang gabi",
        "magandang araw",
    }
}


_GREETING_RESPONSES = {
    "english": [
        "Hello! I'm **KALAW**, your CPSU Faculty Manual assistant.\n\nI can help you with policies, procedures, leave benefits, faculty ranks, and anything covered in the Faculty Manual. What would you like to know?",

        "Hi there! Welcome — I'm **KALAW**, the CPSU Faculty Manual chatbot.\n\nFeel free to ask me about faculty policies, duties, benefits, or any section of the manual. How can I assist you today?",

        "Good day! I'm **KALAW**, here to help you navigate the CPSU Faculty Manual.\n\nAsk me anything about faculty rules, leave policies, promotions, or academic procedures. What's your question?",

        "Hey! I'm **KALAW** — your go-to guide for the CPSU Faculty Manual.\n\nWhether it's about teaching loads, leave policies, or faculty obligations, I'm ready to help. What do you need?"
    ],

    "cebuano": [
        "Maayong adlaw! Ako si **KALAW**, ang imong CPSU Faculty Manual assistant.\n\nMakatabang ko nimo sa mga polisiya, proseso, leave benefits, faculty ranks, ug uban pang impormasyon nga makita sa Faculty Manual. Unsa imong gusto mahibal-an?",

        "Maayong adlaw! 👋 Ako si **KALAW**, imong assistant para sa CPSU Faculty Manual.\n\nPwede ko nimo pangutan-on bahin sa faculty policies, duties, benefits, teaching load, ug uban pang naa sa manual. Unsaon nako pagtabang nimo?",

        "Kumusta! Ako si **KALAW**. Andam ko motabang nimo sa pagkuha og impormasyon gikan sa CPSU Faculty Manual.\n\nUnsa imong gusto pangutan-on?"
    ],

    "filipino": [
        "Magandang araw! Ako si **KALAW**, ang iyong CPSU Faculty Manual assistant.\n\nMaaari kitang tulungan tungkol sa mga polisiya, proseso, leave benefits, faculty ranks, at iba pang impormasyon na nakapaloob sa Faculty Manual. Ano ang nais mong malaman?",

        "Kumusta! 👋 Ako si **KALAW**, ang iyong assistant para sa CPSU Faculty Manual.\n\nMaaari mo akong tanungin tungkol sa faculty policies, duties, benefits, teaching load, at iba pang impormasyon sa manual. Paano kita matutulungan?",

        "Magandang araw! Ako si **KALAW**. Handa akong tumulong sa paghahanap ng impormasyon mula sa CPSU Faculty Manual.\n\nAno ang nais mong itanong?"
    ]
}


def _normalize_greeting(query: str) -> str:
    """Normalize only for greeting detection."""
    q = str(query).lower().replace("’", "'")
    q = re.sub(r"[^a-z0-9\s']", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def greeting_match(query: str) -> str | None:
    """Detect greetings and return a response in the same language."""
    q = _normalize_greeting(query)

    if not q:
        return None

    for language in ("cebuano", "filipino", "english"):
        for trigger in sorted(_GREETING_TRIGGERS[language], key=len, reverse=True):
            if q == trigger or q.startswith(trigger + " "):
                return random.choice(_GREETING_RESPONSES[language])

    return None
