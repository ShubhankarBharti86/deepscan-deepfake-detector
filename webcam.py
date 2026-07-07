"""Live webcam deepfake detection using a trained DeepSCN checkpoint."""
import argparse, cv2, torch, numpy as np
import torch.nn.functional as F
from models.deepscn_model import build_model
from utils.preprocess import get_val_transform, FaceExtractor


def load_model(checkpoint, variant, device):
    model = build_model(variant).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/deepscn_best.pth')
    ap.add_argument('--variant', default='full', choices=['full','lite'])
    ap.add_argument('--camera', type=int, default=0)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, args.variant, device)
    transform = get_val_transform()
    extractor = FaceExtractor(device=device)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam')
    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = extractor.extract(rgb)
        if face is None: face = rgb
        x = transform(face).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = F.softmax(model(x), dim=1)[0,1].item()
        label = 'FAKE' if prob > .5 else 'REAL'
        color = (0,0,255) if label == 'FAKE' else (0,255,0)
        cv2.putText(frame, f'{label} fake={prob*100:.1f}%', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('DeepSCN Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
