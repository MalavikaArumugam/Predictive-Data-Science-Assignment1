import csv
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]/"datasets"/"linear_regression_data.csv"
with DATA.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
x=[float(r["hours"]) for r in rows]; y=[float(r["score"]) for r in rows]; n=len(x); xm=sum(x)/n; ym=sum(y)/n
m=sum((a-xm)*(b-ym) for a,b in zip(x,y))/sum((a-xm)**2 for a in x); c=ym-m*xm
p=[c+m*a for a in x]; r2=1-sum((b-q)**2 for b,q in zip(y,p))/sum((b-ym)**2 for b in y)
print("="*48); print("SIMPLE LINEAR REGRESSION"); print("="*48); print(f"Number of records : {n}"); print(f"Regression equation: Score = {c:.2f} + {m:.2f} × Hours"); print(f"R-squared          : {r2:.3f}"); print(f"Predicted score for 6.5 study hours: {c+m*6.5:.2f}"); print("Note: Illustrative dataset for learning only.")
