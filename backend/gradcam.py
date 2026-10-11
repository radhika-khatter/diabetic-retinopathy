"""Step 5: Grad-CAM on ResNet50 layer4."""
import cv2, numpy as np, torch, torch.nn.functional as F

class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)
    def _fwd(self, m, i, o):
        self.acts = o
        o.register_hook(lambda g: setattr(self, "grads", g))
    def __call__(self, x, cls=None):
        self.model.zero_grad()
        out = self.model(x)
        cls = int(out.argmax(1).item()) if cls is None else cls
        out[0, cls].backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True)).detach()
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.cpu().numpy(), out.detach()

def overlay(rgb, cam, alpha=0.45):
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)
