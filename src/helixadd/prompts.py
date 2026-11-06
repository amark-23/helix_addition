import random

PREFIX = (
    "calc: 31+45=76\n"
    "calc: 17+29=46\n"
    "calc: 52+86=138\n"
    "calc: 63+78=141\n"
)

def make_prompt(a, b, few_shot=True): # make prompt for addition 
    question = f"calc: {a}+{b}="
    if few_shot:
        return PREFIX + question
    return question

def is_carry(a, b):
    return (a % 10) + (b % 10) >= 10

EXCLUDED = {(31, 45), (17, 29), (52, 86), (63, 78)}

def all_pairs(): # returns all pairs (a, b) without excuded pairs
    pairs = []
    for a in range(100):
        for b in range(100):
            # skip (a, b) if it, or its swap (b, a), is in EXCLUDED
            # otherwise add (a, b) to pairs
            if (a,b) in EXCLUDED or (b, a) in EXCLUDED:
                continue
            pairs.append((a, b))
    return pairs

def sample_pairs(n, seed=0):
    rng = random.Random(seed)
    if n % 2 != 0:
       raise ValueError(f"n must be even, got {n}")    
    carry = [(a,b) for a,b in all_pairs() if is_carry(a,b)]
    no_carry = [(a,b) for a,b in all_pairs() if not is_carry(a,b)]
    chosen = rng.sample(carry, n//2) + rng.sample(no_carry, n//2) # half carry, half no carry
    rng.shuffle(chosen)
    return chosen
