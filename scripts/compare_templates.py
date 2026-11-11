"""Clean-run errors under the two prompt templates (few-shot vs zero-shot).

Reads results/clean_accuracy (few-shot) and results/clean_accuracy_zeroshot.
diff = the model's answer - the correct answer.
"""
import pandas as pd

fs = pd.read_csv("results/clean_accuracy/data.csv")
zs = pd.read_csv("results/clean_accuracy_zeroshot/data.csv")
for df in (fs, zs):
    df["diff"] = df["pred"] - (df["a"] + df["b"])

for name, df in [("few-shot", fs), ("zero-shot", zs)]:
    print(f"diff of the wrong answers, {name} prompt:")
    print(df[df["diff"] != 0]["diff"].value_counts().to_string(), "\n")

print("zero-shot wrong answers:")
print(zs[zs["diff"] != 0][["a", "b", "pred", "diff"]].to_string(), "\n")

odd = zs[(zs["diff"] != 0) & (zs["diff"].abs() != 10)]
print("zero-shot wrong answers that are not +-10, with the raw output:")
print(odd[["a", "b", "pred", "diff", "raw"]].to_string())
