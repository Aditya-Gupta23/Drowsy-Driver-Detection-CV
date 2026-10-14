# HeavyDrivers — Driver Drowsiness Detection System

A computer vision-based driver drowsiness detection system that combines deep learning and facial landmark analysis to identify signs of driver fatigue, prolonged eye closure, and possible yawning in real time.

## Overview

Driver drowsiness is a significant road safety concern. HeavyDrivers aims to detect visual indicators of fatigue through a webcam feed and provide an audio alert when potential drowsiness is detected.

The system combines a fine-tuned **MobileNetV3-Small** image classification model with **MediaPipe Face Mesh** for facial landmark analysis. Temporal checks help distinguish brief eye blinks from prolonged eye closure and sustained mouth opening.

## Features

- **Real-time drowsiness classification:** Classifies facial images into `active` and `fatigue` categories.
- **Prolonged eye-closure detection:** Uses eye landmarks and the Eye Aspect Ratio (EAR) to identify sustained eye closure.
- **Possible yawn detection:** Uses mouth landmarks and the Mouth Aspect Ratio (MAR) to detect sustained mouth opening.
- **Temporal analysis:** Evaluates consecutive frames and duration thresholds to reduce false alarms from brief blinks.
- **Audio alerts:** Generates an audible alert when configured fatigue conditions are met.
- **Transfer learning:** Uses an ImageNet-pretrained MobileNetV3-Small model fine-tuned on the UTA-RLDD dataset.
- **Model checkpointing:** Saves the best-performing model based on validation macro-F1.
- **Apple Silicon support:** Supports PyTorch's Metal Performance Shaders (MPS) backend on compatible Macs.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch, Torchvision |
| Image processing and webcam access | OpenCV |
| Facial landmark analysis | MediaPipe |
| Classification model | MobileNetV3-Small |
| Evaluation metrics | Scikit-learn |
| Training hardware acceleration | Apple Silicon MPS, when available |

## System Architecture

```text
Webcam Feed
    |
    v
Frame Capture (OpenCV)
    |
    +------------------------------+
    |                              |
    v                              v
MobileNetV3-Small             MediaPipe Face Mesh
    |                              |
    v                              +----------------------+
Active / Fatigue                    |                     |
Classification                     v                     v
                              Eye Landmarks         Mouth Landmarks
                                   |                     |
                                   v                     v
                              EAR Analysis           MAR Analysis
                                   |                     |
                                   +----------+----------+
                                              |
                                              v
                                    Temporal Condition Checks
                                              |
                                              v
                                       Fatigue Decision
                                              |
                                              v
                                         Audio Alert
```

The classification model and facial landmark checks provide complementary signals. The webcam application uses configurable thresholds and temporal conditions to determine when to display warnings or trigger an alert.

## Dataset

The project uses the **UTA-RLDD (Real-Life Drowsiness Dataset)**.

- Kaggle dataset: https://www.kaggle.com/datasets/minhngt02/uta-rldd
- Dataset project page: https://sites.google.com/view/utarldd/home

The dataset is organized into training, validation, and test splits using manifest files.

| Split | Active | Fatigue | Total |
|---|---:|---:|---:|
| Training | 3,192 | 3,192 | 6,384 |
| Validation | 912 | 803 | 1,715 |
| Test | 456 | 453 | 909 |
| **Total** | **4,560** | **4,448** | **9,008** |

**Class mapping:**
- `active` → `0`
- `fatigue` → `1`

The training split was balanced by using 3,192 images per class. The validation and test splits retain their respective class distributions.

The reported counts refer to images listed in the project's manifest files, not necessarily every image present in the extracted dataset folders.

## Model and Training

### Model

The classification component uses **MobileNetV3-Small**, initialized with pretrained ImageNet weights and fine-tuned for binary classification.

The final classification layer is replaced to predict two classes:

- Active
- Fatigue

### Training configuration

| Parameter | Value |
|---|---|
| Input image size | 224 × 224 |
| Batch size | 16 |
| Maximum epochs | 25 |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Loss function | Cross-entropy loss |
| Learning-rate scheduler | ReduceLROnPlateau |
| Early stopping | Enabled |
| Model-selection metric | Validation macro-F1 |
| Normalization | ImageNet mean and standard deviation |
| Device | Apple MPS when available, otherwise CPU |

Training-time augmentation includes random horizontal flipping, small random rotations, and brightness/contrast adjustments.

The training script saves the best checkpoint based on validation macro-F1 rather than simply using the last epoch. Training can stop early when validation performance no longer improves sufficiently.

## Results

The best checkpoint was selected at **epoch 16** based on validation macro-F1.

### Test-set performance

| Metric | Active | Fatigue |
|---|---:|---:|
| Precision | 0.96 | 0.96 |
| Recall | 0.96 | 0.96 |
| F1-score | 0.96 | 0.96 |
| Test samples | 456 | 453 |

| Overall metric | Result |
|---|---:|
| Test accuracy | 95.82% |
| Macro-F1 | 0.9582 |
| Test loss | 0.1110 |

These results describe performance on the project's specified test split. They should not be interpreted as a guarantee of real-world driver monitoring accuracy.

## Installation

### Prerequisites

- Python 3.10
- A webcam for real-time testing
- Git, if cloning the repository
- Apple Silicon Mac with MPS support for accelerated training, or a compatible CPU environment

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd HeavyDrivers
```

Replace `<YOUR_REPOSITORY_URL>` with your repository's Git URL.

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Ensure `requirements.txt` is present in the project root.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The provided dependency configuration targets Python 3.10 and includes PyTorch, Torchvision, NumPy, OpenCV, MediaPipe, Pillow, and Scikit-learn.

**Important:** Avoid installing `opencv-python` alongside `opencv-contrib-python`, as both packages provide the `cv2` module.

## Project Structure

```text
HeavyDrivers/
├── archive/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── active/
│   └── fatigue/
├── model_artifacts/
│   ├── drowsiness_model_25ep_best.pth
│   ├── config_25ep.json
│   └── training_history_25ep.json
├── train_25_epochs_early_stopping.py
├── webcam_test.py
├── requirements.txt
└── README.md
```

The exact directory structure may vary depending on how the dataset and scripts are organized locally.

### Important files

- `train_25_epochs_early_stopping.py` — trains and evaluates the classification model.
- `webcam_test.py` — runs real-time webcam inference and facial landmark checks.
- `model_artifacts/drowsiness_model_25ep_best.pth` — best saved model checkpoint.
- `model_artifacts/config_25ep.json` — model configuration used by the application.
- `model_artifacts/training_history_25ep.json` — training history recorded by the training script.
- `requirements.txt` — Python dependencies.

## Running the Project

### Train the model

First, verify that the dataset manifests and image paths match the locations expected by the training script.

Then run:

```bash
python train_25_epochs_early_stopping.py
```

The script trains the model, evaluates validation performance, saves the best checkpoint, records training history, and evaluates the selected model on the test split.

If the best checkpoint already exists, back it up before running another training session if you want to preserve the current results.

### Run real-time detection

After confirming that the model checkpoint and configuration file exist, start the webcam application:

```bash
python webcam_test.py
```

Allow camera access if macOS requests permission. Ensure that the model and configuration paths in `webcam_test.py` point to the correct files.

For example:

```python
MODEL_PATH = MODEL_DIR / "drowsiness_model_25ep_best.pth"
CONFIG_PATH = MODEL_DIR / "config_25ep.json"
```

The configuration filename must match the file actually present in `model_artifacts/`. If your local configuration file has a different name, update the path accordingly.

### Detection logic

**1. Image classification**

MobileNetV3-Small predicts whether the input face image belongs to the active or fatigue class. Predictions are evaluated over a rolling window to reduce the influence of individual frames.

**2. Eye-closure detection**

The Eye Aspect Ratio (EAR) is calculated from selected eye landmarks. A sustained reduction below the configured threshold can trigger a prolonged-eye-closure warning.

**3. Yawn detection**

The Mouth Aspect Ratio (MAR) is calculated from mouth landmarks. Sustained mouth opening above the configured threshold triggers a possible-yawn warning.

**4. Audio alert**

When the configured detection conditions are satisfied, the application can generate an audio alert. Thresholds, durations, and cooldown settings can be adjusted in the webcam script.

## Limitations

- The system relies on visible facial features and may be affected by lighting, camera position, occlusion, glasses, and face orientation.
- Classification performance depends on the dataset and may differ across drivers and real-world conditions.
- Eye blinks, facial expressions, and speech can resemble fatigue-related visual signals.
- A possible yawn is not definitive evidence of drowsiness.
- Validation and test scores do not establish safety-critical reliability in real driving conditions.
- The test results should be interpreted with caution unless the split is verified to be independent by driver or recording session.
- This is an experimental computer vision project, not a certified automotive safety system.

**Do not rely on this prototype as the sole means of detecting driver fatigue or preventing accidents.**

## Future Improvements

- Evaluate on driver-independent and more diverse datasets.
- Test performance under different lighting and camera conditions.
- Tune EAR, MAR, and classification thresholds using a separate validation protocol.
- Track precision, recall, false alarms, and missed detections during real-time tests.
- Improve temporal fusion of classification scores and facial landmark measurements.
- Explore additional cues, such as head pose, for more robust fatigue analysis.
- Test performance on additional hardware and optimize inference latency.

## References

1. UTA-RLDD dataset: https://www.kaggle.com/datasets/minhngt02/uta-rldd
2. UTA-RLDD project page: https://sites.google.com/view/utarldd/home
3. MobileNetV3 paper: https://arxiv.org/abs/1905.02244
4. Torchvision MobileNetV3-Small documentation: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html
5. MediaPipe Face Mesh documentation: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md


