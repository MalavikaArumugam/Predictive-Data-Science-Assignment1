import csv
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]/"datasets"/"classification_data.csv"
with DATA.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
features=["hours_studied","attendance"]; X=[[float(r[k]) for k in features] for r in rows]; y=[int(r["passed"]) for r in rows]
def g(a):
 if not a:return 0
 p=sum(a)/len(a); return 1-p*p-(1-p)*(1-p)
best=None;base=g(y)
for j,name in enumerate(features):
 vals=sorted(set(r[j] for r in X))
 for a,b in zip(vals,vals[1:]):
  t=(a+b)/2; L=[q for r,q in zip(X,y) if r[j]<=t]; R=[q for r,q in zip(X,y) if r[j]>t]; gain=base-len(L)/len(y)*g(L)-len(R)/len(y)*g(R)
  if best is None or gain>best[0]:best=(gain,name,j,t)
_,name,j,t=best; L=[q for r,q in zip(X,y) if r[j]<=t]; R=[q for r,q in zip(X,y) if r[j]>t]; lc=int(sum(L)>=len(L)/2); rc=int(sum(R)>=len(R)/2); pred=[lc if r[j]<=t else rc for r in X]; acc=sum(a==b for a,b in zip(pred,y))/len(y); result=lc if {"hours_studied":5.,"attendance":82.}[name]<=t else rc
print("="*48); print("DECISION TREE CLASSIFIER"); print("="*48); print(f"Chosen split       : {name} <= {t:.2f}"); print(f"Training accuracy  : {acc*100:.1f}%"); print(f"Prediction (5 study hours, 82% attendance): {'PASS' if result else 'FAIL'}"); print("Note: Deliberately small, one-split teaching tree.")
