import csv
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]/"datasets"/"classification_data.csv"
with DATA.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
raw=[[float(r["hours_studied"]),float(r["attendance"])/10] for r in rows]; y=[1 if int(r["passed"]) else -1 for r in rows]; mu=[sum(r[j] for r in raw)/len(raw) for j in range(2)]; sd=[(sum((r[j]-mu[j])**2 for r in raw)/len(raw))**.5 or 1 for j in range(2)]; X=[[(r[j]-mu[j])/sd[j] for j in range(2)] for r in raw]; w=[0.,0.]; b=0.
for _ in range(5000):
 for r,t in zip(X,y):
  if t*(w[0]*r[0]+w[1]*r[1]+b)<1: w[0]+=.01*(t*r[0]-2*w[0]); w[1]+=.01*(t*r[1]-2*w[1]); b+=.01*t
  else: w[0]-=.02*w[0]; w[1]-=.02*w[1]
def pred(r): z=[(r[j]-mu[j])/sd[j] for j in range(2)]; return 1 if w[0]*z[0]+w[1]*z[1]+b>=0 else -1
acc=sum(pred(r)==t for r,t in zip(raw,y))/len(y); result=pred([5.,8.2]); print("="*48); print("LINEAR SVM CLASSIFIER"); print("="*48); print(f"Number of records : {len(y)}"); print(f"Training accuracy : {acc*100:.1f}%"); print("Prediction (5 study hours, 82% attendance):", "PASS" if result==1 else "FAIL"); print("Note: Simple hinge-loss gradient descent teaching implementation.")
