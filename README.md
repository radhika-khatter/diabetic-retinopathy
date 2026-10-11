# Retinova — AI-Based Diabetic Retinopathy Screening

Retinova is an AI-powered retinal screening application that uses deep learning to classify diabetic retinopathy severity from fundus images. It combines image quality assessment, explainable AI, confidence calibration, and referral recommendations in a web-based interface.

## Key Features

- **Five-Class Classification:** Predicts diabetic retinopathy severity using a ResNet50 deep learning model.
- **Image Quality Assessment:** Evaluates retinal images for issues such as blur, brightness, and contrast.
- **Image Preprocessing:** Applies retinal image processing techniques, including cropping, resizing, and CLAHE enhancement.
- **Explainable AI:** Uses Grad-CAM to visualize image regions influencing model predictions.
- **Confidence Calibration:** Applies temperature scaling to calibrate model confidence.
- **Referral Support:** Provides screening recommendations based on the predicted grade, confidence, and image quality.
- **Web Interface:** Allows users to upload fundus images and view screening results through a React frontend connected to a FastAPI backend.

## Model Performance

The model was evaluated on a held-out test split of the APTOS 2019 dataset.

| Metric | Result |
|---|---:|
| Five-class classification accuracy | 80.18% |
| Quadratic Weighted Kappa (QWK) | 0.8784 |
| ROC-AUC for referable diabetic retinopathy | 98.66% |
| Sensitivity for referable DR | 91.93% |
| Specificity for referable DR | 94.19% |

Referable diabetic retinopathy is defined as grades 2–4. Sensitivity and specificity were evaluated using the selected referral threshold.

These results reflect the project's held-out test split and do not establish clinical validity or performance on other populations.

## Technology Stack

- **Programming:** Python, JavaScript
- **Deep Learning:** PyTorch, Torchvision, ResNet50
- **Image Processing:** OpenCV, NumPy
- **Machine Learning:** Scikit-learn, Pandas
- **Backend:** FastAPI, Uvicorn
- **Frontend:** React, Vite, Axios
- **Dataset:** APTOS 2019 Blindness Detection

## Project Structure

```text
retinova/
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── preprocess.py
│   ├── quality.py
│   ├── gradcam.py
│   ├── calibrate.py
│   ├── referral.py
│   ├── train.py
│   └── drishti.pt
├── frontend/
│   ├── src/
│   ├── public/
│   ├── index.html
│   └── package.json
├── data/
├── requirements.txt
├── .gitignore
└── README.md
```

The saved model checkpoint is named `drishti.pt` in the current implementation. The filename can remain unchanged even though the application is now called Retinova.

## Installation and Setup

### Prerequisites

- Python 3.11
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/radhika-khatter/diabetic-retinopathy.git
cd diabetic-retinopathy
```

Replace the repository URL if Retinova is hosted in a separate GitHub repository.

### 2. Set up the Python environment

On Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the backend

```powershell
cd backend
python -m uvicorn app:app --reload
```

The API will be available at:

- API: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Keep this terminal running.

### 4. Run the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm install axios
npm run dev
```

Open the local URL printed by Vite, usually:

`http://localhost:5173`

Keep both servers running while using the application.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Checks backend status and returns the saved calibration temperature |
| POST | `/api/analyze` | Accepts a fundus image and returns the screening analysis |

The analysis response includes the predicted grade, confidence, class probabilities, image quality assessment, referral recommendation, Grad-CAM visualization, and preprocessed image.

## Dataset

The model was developed using the **APTOS 2019 Blindness Detection** dataset. The dataset contains retinal fundus images labeled with five diabetic retinopathy severity grades:

| Grade | Severity |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

Dataset access: [APTOS 2019 Blindness Detection — Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

Follow the dataset's terms and conditions when downloading or using the images.

## Current Scope and Future Work

Potential extensions include:

- Lesion segmentation using U-Net and annotated retinal datasets.
- External evaluation on independent retinal image datasets.
- Detailed screening reports and probability visualizations.
- Persistent storage of screening records.
- Rural screening workflow simulation and referral workload analysis.

These are future extensions, not features claimed as complete in the current implementation.

## Disclaimer

Retinova is an academic AI research prototype intended for educational and research purposes. It is not a substitute for professional ophthalmic examination, and its predictions should not be used as the sole basis for clinical decisions. Clinical validation is required before real-world medical use.

## Author

**Radhika Khatter**

- GitHub: [radhika-khatter](https://github.com/radhika-khatter)

