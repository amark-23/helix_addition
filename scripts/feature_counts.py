import numpy as np

d = np.load("results/eq_activations/data.npz")
uid = d["layer"].astype(np.int64) * 131072 + d["feature"]
ids, counts = np.unique(uid, return_counts=True)

print("distinct features firing at '=':", len(ids))
for k in [1, 10, 100, 1000, 5000, 9000]:
    print(f"fire on >= {k:>5} prompts: {(counts >= k).sum()}")

print("\nlayer  distinct  fire on >= 100 prompts")
layer_of = ids // 131072
for L in range(16):
    print(f"{L:>5}  {(layer_of == L).sum():>8}  {((layer_of == L) & (counts >= 100)).sum():>8}")