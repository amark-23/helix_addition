import pytest
from helixadd.readout import outcome, parse_answer


@pytest.mark.parametrize("text, expected", [
    ("95", 95),            # plain answer
    ("95\n", 95),          # answer, then newline
    (" 95", 95),           # leading space is skipped
    ("138abc", 138),       # 3-digit answer, junk after it
    ("9", 9),              # single digit
    ("the", None),         # no number: X
    ("", None),            # empty output: X
    ("95+3", 95),          # stops at the first non-digit
    ("9 5", 9),            # a space ends the number
    ("12\n34", 12),        # a newline ends the number
    ("abc95", None),       # number not at the start: X
    ("²", None),           # superscript 2 is not an ordinary digit: X
])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


@pytest.mark.parametrize("pred, correct, expected", [
    (95, 95, "N"),
    (92, 95, "U"),            # units only
    (85, 95, "M"),            # upper only
    (82, 95, "B"),            # both: e.g. the carry lost with the units
    (105, 95, "M"),           # 9 -> 10 in the upper part, units the same
    (195, 95, "M"),
    (9, 95, "B"),             # a single digit read as the answer
    (None, 95, "X"),
    (float("nan"), 95, "X"),
    (7, 7, "N"),              # single-digit answers: upper part 0
    (17, 7, "M"),
])
def test_outcome(pred, correct, expected):
    assert outcome(pred, correct) == expected
