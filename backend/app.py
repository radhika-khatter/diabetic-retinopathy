"""FastAPI backend: Quality -> Preprocess -> ResNet50 -> Calibration -> Grad-CAM -> Referral.
Run: uvicorn app:app --reload   then open http://127.0.0.1:8000/docs
"""
import base64, cv2, numpy as np, torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms as T
from model import build_model, MEAN, STD
from quality import assess_quality
from preprocess import preprocess
from gradcam import GradCAM, overlay
from calibrate import softmax
from referral import decide, GRADES

app = FastAPI(title="RETINOVA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
ck = torch.load("retinova.pt", map_location="cpu")
model = build_model(pretrained=False); model.load_state_dict(ck["model"]); model.eval()
cam_engine = GradCAM(model, model.layer4)
tf = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(MEAN, STD)])

def b64(rgb):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1]).decode()

@app.get("/api/health")
def health(): return {"status": "ok", "temperature": ck["T"]}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    q = assess_quality(img)
    rgb = preprocess(img)
    x = tf(rgb).unsqueeze(0)
    cam, logits = cam_engine(x)
    probs = softmax(logits.numpy(), ck["T"])[0]
    grade, conf = int(probs.argmax()), float(probs.max())
    action, reasons = decide(grade, conf, q["ok"])
    return {"quality": q, "grade": grade, "grade_name": GRADES[grade], "confidence": conf,
            "probabilities": {g: float(p) for g, p in zip(GRADES, probs)},
            "referable_prob": float(probs[2:].sum()),
            "action": action, "reasons": reasons,
            "gradcam": b64(overlay(rgb, cam)), "preprocessed": b64(rgb)}
