# Retinova - mid-project prototype
Pipeline: Quality -> Preprocess (CLAHE) -> ResNet50 -> Temperature scaling -> Grad-CAM -> Referral -> JSON report

## Setup
pip install -r requirements.txt
Download APTOS 2019 from Kaggle (aptos2019-blindness-detection) into data/aptos/:
  data/aptos/train.csv  and  data/aptos/train_images/*.png

## Train + evaluate (writes results.json and retinova.pt)
cd backend
python train.py --epochs 8            # quick test: add --limit 600

## Run API
uvicorn app:app --reload   ->   http://127.0.0.1:8000/docs   (upload a fundus image)

## Not done yet (weeks 5-12)
U-Net lesion segmentation (IDRiD), React UI, MongoDB, external validation (Messidor-2),
rural queue simulation, tuning of quality thresholds.
