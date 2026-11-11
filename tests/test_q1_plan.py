from helixadd import q1


def test_plan():
    runs = q1.plan()
    assert len(runs) == 10 + 6 * q1.N_RANDOM
    assert len(runs["clean"]) == 0
    for name in q1.COMPARED:
        group = runs[name]
        for k in (0, q1.N_RANDOM - 1):
            r = runs[f"random | {name} | {k:03d}"]
            assert len(r) == len(group)
            assert len(r.merge(group)) == 0                      # never the group's own features
    assert len({q1.file_name(n) for n in runs}) == len(runs)      # one file per run


def test_eval_pairs():
    pairs = q1.eval_pairs()
    assert 990 <= len(pairs) <= 1000                              # 1,000 minus the clean-wrong ones
    assert len(set(pairs)) == len(pairs)
