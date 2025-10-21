from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def ensure_outdir(p):
 p=Path(p); p.mkdir(parents=True, exist_ok=True); return p

def save_csv(df,p):
 p=Path(p); p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p,index=False); return p

def plot_roc(fpr,tpr,out):
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(6,6)); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],'--'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC'); fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig)
