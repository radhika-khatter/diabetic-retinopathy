"""Step 1: Quality assessment (blur, brightness/contrast, field of view)."""
import cv2, numpy as np

def assess_quality(img_bgr, blur_thr=60.0, bright=(40, 200), contrast_thr=20.0, fov_thr=0.40):
    img = cv2.resize(img_bgr, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    fov_mask = gray > 15
    fov = float(fov_mask.mean())
    px = gray[fov_mask] if fov_mask.any() else gray.ravel()
    brightness, contrast = float(px.mean()), float(px.std())
    issues = []
    if blur < blur_thr: issues.append("Image is blurred / out of focus")
    if brightness < bright[0]: issues.append("Image too dark")
    if brightness > bright[1]: issues.append("Image overexposed")
    if contrast < contrast_thr: issues.append("Low contrast")
    if fov < fov_thr: issues.append("Retina not fully in field of view")
    return {"ok": len(issues) == 0, "issues": issues,
            "blur": blur, "brightness": brightness, "contrast": contrast, "fov": fov}
