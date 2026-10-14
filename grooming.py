"""SafeChat grooming screener (v1) — transparent, rule-based red-flag check.

How it works: scans a conversation for known predator warning patterns
(secrecy, meeting requests, gifts, personal-info requests, love-bombing,
moving to private apps). Each matched pattern belongs to a category;
risk = number of distinct categories matched.

This is a heuristic early-warning screener, NOT a trained ML model.
It is honest about that: every flag it raises is shown to the user with
the exact phrase that triggered it.
"""

# (human-friendly category name, explanation shown to the child, trigger phrases)
RED_FLAGS = [
    ("Secrecy",
     "They ask you to keep things secret — a classic warning sign.",
     ["don't tell your parents", "dont tell your parents", "keep it secret",
      "keep this secret", "our secret", "delete this chat", "delete our chats",
      "don't tell anyone", "dont tell anyone", "this stays between us"]),
    ("Meeting in person",
     "They try to meet you alone in real life.",
     ["meet me", "come over", "meet up", "see you alone", "come to my place",
      "meet in person", "pick you up"]),
    ("Gifts and bribes",
     "They offer gifts, money, or treats to win your trust.",
     ["buy you", "gift for you", "i'll give you", "ill give you",
      "free phone", "give you money", "treat for you"]),
    ("Asking personal info",
     "They ask for private things like photos, address, or school.",
     ["send me your photo", "send me a photo", "send pics", "send nudes",
      "where do you live", "what school do you", "your address",
      "phone number", "send me your pic"]),
    ("Love-bombing",
     "A stranger showers you with affection to gain trust fast.",
     ["you're special", "youre special", "i love you", "you're so mature",
      "youre so mature", "you're different", "youre different",
      "you understand me"]),
    ("Moving to private apps",
     "They try to move the chat somewhere private, away from others.",
     ["add me on snapchat", "snapchat me", "whatsapp me", "talk on whatsapp",
      "instagram dm", "private chat", "talk privately", "download telegram"]),
]

def screen_conversation(text):
    """Returns (risk_level, matched_flags).
    risk_level: 'low' | 'medium' | 'high'
    matched_flags: list of (category, explanation, trigger_phrase)
    """
    low = text.lower()
    matched = []
    seen_categories = set()
    for category, explanation, phrases in RED_FLAGS:
        for phrase in phrases:
            if phrase in low and category not in seen_categories:
                matched.append((category, explanation, phrase))
                seen_categories.add(category)
                break  # one flag per category is enough
    n = len(matched)
    if n == 0:
        risk = "low"
    elif n == 1:
        risk = "medium"
    else:
        risk = "high"
    return risk, matched


RISK_TEXT = {
    "low": ("Looks okay", "No warning signs found in this conversation. Stay careful anyway — "
            "never share photos, address, or school with strangers.", "green"),
    "medium": ("One warning sign", "This conversation has one warning sign. Be careful: don't share "
               "anything personal, and consider telling a trusted adult.", "orange"),
    "high": ("Multiple warning signs — be very careful",
             "This conversation shows MULTIPLE warning signs used by online predators.", "red"),
}

GROOMING_GUIDANCE = [
    "Do NOT meet this person — not today, not ever, not even in public.",
    "Do NOT send photos, your address, school name, or phone number.",
    "Tell a trusted adult RIGHT NOW — a parent, teacher, or counsellor.",
    "Save the chat as evidence (screenshots).",
    "Block the person after an adult knows.",
]
