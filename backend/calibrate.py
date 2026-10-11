"""Step 6: Temperature scaling + ECE."""
import numpy as np, torch, torch.nn.functional as F

def fit_temperature(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.05), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.clamp(min=0.05).item())

def softmax(logits, T=1.0):
    z = np.asarray(logits) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def ece(probs, labels, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)
