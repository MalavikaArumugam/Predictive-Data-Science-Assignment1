import csv
from itertools import combinations
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]/"datasets"/"market_basket.csv"
with DATA.open(newline="",encoding="utf-8") as f: T=[set(r["items"].split(";")) for r in csv.DictReader(f)]
n=len(T); items=sorted(set().union(*T)); support=lambda s:sum(s.issubset(t) for t in T)/n; ms=.4; mc=.6
print("="*58); print("MARKET BASKET ANALYSIS — ASSOCIATION RULES"); print("="*58); print(f"Number of transactions: {n}"); print(f"Minimum support: {ms:.2f}"); print(f"Minimum confidence: {mc:.2f}"); F=[]
for k in range(1,4):
 for c in combinations(items,k):
  s=support(set(c))
  if s>=ms:F.append((set(c),s)); print(f"Frequent itemset {sorted(c)}: support = {s:.2f}")
print("\nAssociation rules:")
for I,s in F:
 if len(I)<2:continue
 a=sorted(I)
 for k in range(1,len(a)):
  for lt in combinations(a,k):
   L=set(lt); R=I-L; conf=s/support(L); lift=conf/support(R)
   if conf>=mc: print(f"{sorted(L)} -> {sorted(R)} | confidence = {conf:.2f}, lift = {lift:.2f}")
print("Interpretation: lift > 1 indicates positive association in this sample."); print("Note: Tiny illustrative dataset, not a business recommendation.")
