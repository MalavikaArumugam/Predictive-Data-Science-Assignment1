import csv,math
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]/"datasets"/"classification_data.csv"
with DATA.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
X=[[float(r["hours_studied"]),float(r["attendance"])] for r in rows]; y=[int(r["passed"]) for r in rows]; classes=sorted(set(y)); stats={}
for c in classes:
 cr=[r for r,t in zip(X,y) if t==c]; stats[c]=[]
 for j in range(2):
  v=[r[j] for r in cr]; m=sum(v)/len(v); var=sum((q-m)**2 for q in v)/len(v); stats[c].append((m,max(var,1e-6)))
pri={c:y.count(c)/len(y) for c in classes}
def predict(r):
 scores={}
 for c in classes:
  s=math.log(pri[c])
  for j,v in enumerate(r): m,var=stats[c][j]; s+=-.5*math.log(2*math.pi*var)-(v-m)**2/(2*var)
  scores[c]=s
 return max(scores,key=scores.get)
p=[predict(r) for r in X]; acc=sum(a==b for a,b in zip(p,y))/len(y); result=predict([5.,82.]); print("="*48); print("GAUSSIAN NAIVE BAYES"); print("="*48); print(f"Number of records : {len(y)}"); print(f"Training accuracy : {acc*100:.1f}%"); print(f"Prediction (5 study hours, 82% attendance): {'PASS' if result else 'FAIL'}"); print("Note: Illustrative dataset; training accuracy is not a test-set score.")
