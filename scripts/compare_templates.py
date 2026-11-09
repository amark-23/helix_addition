#Load both CSVs. pd.read_csv("results/clean_accuracy/data.csv") gives back a DataFrame
#For each one, add the diff column.
#Print the value_counts() of diff for the wrong rows of each.
#Print the zero-shot wrong rows with a, b, pred and diff.

import pandas as pd

fs = pd.read_csv("results/clean_accuracy/data.csv")
zs = pd.read_csv("results/clean_accuracy_zeroshot/data.csv")

#diff  =model answer -correct answer
fs['diff'] = fs['pred'] - (fs['a'] + fs['b'])
zs['diff'] = zs['pred'] - (zs['a'] + zs['b'])


print("Value counts of diff for wrong rows in few-shot model:")   
print(fs[fs['diff'] != 0]['diff'].value_counts())

print("Value counts of diff for wrong rows in zero-shot model:")
print(zs[zs['diff'] != 0]['diff'].value_counts())   

#print the zero-shot wrong rows with only a, b, pred and diff, using the double-bracket column selection.
print("Zero-shot wrong rows with a, b, pred and diff:")
print(zs[zs['diff'] != 0][['a', 'b', 'pred', 'diff']])

odd = zs[(zs['diff'] != 0) & (zs['diff'].abs() != 10)]
print("odd outputs")
print(odd[['a', 'b', 'pred', 'diff', 'raw']])
