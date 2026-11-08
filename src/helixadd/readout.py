

def parse_answer(text):
    text = text.lstrip()
    digits = []
    for ch in text:
        if ch in "0123456789":
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None