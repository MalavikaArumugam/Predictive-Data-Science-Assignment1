import csv,math
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]/"datasets"/"classification_data.csv"
with DATA.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
X=[[float(r["hours_studied"]),float(r["attendance"])/10] for r in rows]; y=[int(r["passed"]) for r in rows]; mu=[sum(r[j] for r in X)/len(X) for j in range(2)]; sd=[]
for j in range(2): sd.append((sum((r[j]-mu[j])**2 for r in X)/len(X))**.5 or 1)
Z=[[(r[j]-mu[j])/sd[j] for j in range(2)] for r in X]; w=[0.,0.,0.]
def sig(z): return 1/(1+math.exp(-max(-30,min(30,z))))
for _ in range(4000):
 g=[0.,0.,0.]
 for r,t in zip(Z,y):
  p=sig(w[0]+w[1]*r[0]+w[2]*r[1]); e=p-t; g[0]+=e; g[1]+=e*r[0]; g[2]+=e*r[1]
 w=[a-.1*b/len(y) for a,b in zip(w,g)]
acc=sum((sig(w[0]+w[1]*r[0]+w[2]*r[1])>=.5)==bool(t) for r,t in zip(Z,y))/len(y); z=[(v-mu[j])/sd[j] for j,v in enumerate([6.,8.8])]; p=sig(w[0]+w[1]*z[0]+w[2]*z[1])
print("="*48); print("LOGISTIC REGRESSION"); print("="*48); print(f"Number of records : {len(y)}"); print(f"Training accuracy : {acc*100:.1f}%"); print(f"Predicted pass probability (6 hours, 88% attendance): {p:.3f}"); print("Predicted class    :", "PASS" if p>=.5 else "FAIL"); print("Note: Illustrative dataset; training accuracy is not a test-set score.")
