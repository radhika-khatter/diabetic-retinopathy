"""Step 3: ResNet50 5-class training on APTOS 2019 + real evaluation.
Run from backend/:  python train.py --epochs 8
Expects ../data/aptos/train.csv (id_code,diagnosis) and ../data/aptos/train_images/*.png
"""
import argparse, json, cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score, roc_curve
from preprocess import preprocess
from model import build_model, MEAN, STD
from calibrate import fit_temperature, softmax, ece

p = argparse.ArgumentParser()
p.add_argument("--data", default="../data/aptos"); p.add_argument("--epochs", type=int, default=8)
p.add_argument("--bs", type=int, default=32); p.add_argument("--limit", type=int, default=0)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(f"{a.data}/train.csv")
if a.limit: df = df.sample(a.limit, random_state=1)
tr, rest = train_test_split(df, test_size=0.3, stratify=df.diagnosis, random_state=1)
va, te = train_test_split(rest, test_size=0.5, stratify=rest.diagnosis, random_state=1)
print("Train/Val/Test:", len(tr), len(va), len(te)); print(tr.diagnosis.value_counts().sort_index())

aug = T.Compose([T.ToPILImage(), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                 T.RandomRotation(20), T.ColorJitter(0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
plain = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(MEAN, STD)])

class DS(Dataset):
    def __init__(s, d, tf): s.d, s.tf = d.reset_index(drop=True), tf
    def __len__(s): return len(s.d)
    def __getitem__(s, i):
        r = s.d.iloc[i]
        img = cv2.imread(f"{a.data}/train_images/{r.id_code}.png")
        return s.tf(preprocess(img)), int(r.diagnosis)

mk = lambda d, tf, sh: DataLoader(DS(d, tf), batch_size=a.bs, shuffle=sh, num_workers=0)
Ltr, Lva, Lte = mk(tr, aug, True), mk(va, plain, False), mk(te, plain, False)

counts = tr.diagnosis.value_counts().sort_index().values
w = torch.tensor(len(tr) / (5 * counts), dtype=torch.float32).to(dev)   # class imbalance
model = build_model().to(dev)
crit, opt = nn.CrossEntropyLoss(weight=w), torch.optim.Adam(model.parameters(), lr=1e-4)

@torch.no_grad()
def logits_of(loader):
    model.eval(); L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(dev)).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(L), np.concatenate(Y)

best = -1
for ep in range(a.epochs):
    model.train(); tot = 0
    for x, y in Ltr:
        opt.zero_grad(); loss = crit(model(x.to(dev)), y.to(dev)); loss.backward(); opt.step(); tot += loss.item()
    lv, yv = logits_of(Lva)
    k = cohen_kappa_score(yv, lv.argmax(1), weights="quadratic")
    print(f"epoch {ep+1}: loss {tot/len(Ltr):.3f} | val acc {(lv.argmax(1)==yv).mean():.3f} | val QWK {k:.3f}")
    if k > best: best = k; torch.save(model.state_dict(), "best.pt")

model.load_state_dict(torch.load("best.pt", map_location=dev))
lv, yv = logits_of(Lva); lt, yt = logits_of(Lte)
Tm = fit_temperature(lv, yv)
print(f"\nTemperature = {Tm:.3f} | ECE test before {ece(softmax(lt), yt):.3f} -> after {ece(softmax(lt, Tm), yt):.3f}")

# referable DR (grade >= 2): choose threshold on VAL for ~90% sensitivity, report on TEST
ref = lambda y: (y >= 2).astype(int)
sc = lambda l: softmax(l, Tm)[:, 2:].sum(1)
fpr, tpr, thr = roc_curve(ref(yv), sc(lv)); t = thr[np.argmax(tpr >= 0.90)]
pred = (sc(lt) >= t).astype(int)
tn, fp, fn, tp = confusion_matrix(ref(yt), pred).ravel()
res = {"referable_sensitivity": tp/(tp+fn), "referable_specificity": tn/(tn+fp),
       "referable_AUC": roc_auc_score(ref(yt), sc(lt)), "threshold": float(t),
       "5class_accuracy": float((lt.argmax(1) == yt).mean()),
       "5class_QWK": cohen_kappa_score(yt, lt.argmax(1), weights="quadratic"),
       "confusion_5class": confusion_matrix(yt, lt.argmax(1)).tolist()}
print(json.dumps(res, indent=2))
json.dump(res, open("results.json", "w"), indent=2)
torch.save({"model": model.state_dict(), "T": Tm, "ref_threshold": float(t)}, "retinova.pt")
