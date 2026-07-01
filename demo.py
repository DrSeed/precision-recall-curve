import os,numpy as np,matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve,average_precision_score
os.makedirs("figures",exist_ok=True);os.makedirs("results",exist_ok=True)
rng=np.random.default_rng(6);n=2000;y=(rng.uniform(0,1,n)<0.08).astype(int)  # 8% positives
X=rng.normal(0,1,(n,10));X[y==1,:3]+=1.5
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.4,random_state=0,stratify=y)
clf=LogisticRegression(max_iter=500).fit(Xtr,ytr)
pr,rc,_=precision_recall_curve(yte,clf.predict_proba(Xte)[:,1]);ap=average_precision_score(yte,clf.predict_proba(Xte)[:,1])
plt.figure(figsize=(5,5));plt.plot(rc,pr,label=f"AP={ap:.2f}")
plt.axhline(yte.mean(),ls="--",c="grey",label=f"baseline {yte.mean():.2f}")
plt.xlabel("recall");plt.ylabel("precision");plt.title("Precision-recall curve (demo data)");plt.legend()
plt.tight_layout();plt.savefig("figures/demo.png",dpi=150)
open("results/summary.txt","w").write(f"AP={ap:.3f}\n");print("ok")