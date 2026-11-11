

def parse_answer(text):
    text = text.lstrip()
    digits = []
    for ch in text:
        if ch in "0123456789":
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None


def outcome(pred, correct):
    """Outcome category of one answer (preregistration section 5).

    units = answer mod 10, upper = answer // 10 (95 -> 9, 126 -> 12).
    N: nothing changed, U: units only, M: upper only, B: both, X: no number.
    """
    if pred is None or pred != pred:          # pred != pred is True only for NaN (e.g. read back from a CSV)
        return "X"
    pred = int(pred)
    units_changed = pred % 10 != correct % 10
    upper_changed = pred // 10 != correct // 10
    if units_changed and upper_changed:
        return "B"
    if units_changed:
        return "U"
    if upper_changed:
        return "M"
    return "N"
