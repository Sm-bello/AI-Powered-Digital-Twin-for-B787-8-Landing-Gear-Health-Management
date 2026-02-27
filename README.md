<div align="center">

# ◈ AEROTWIN
### AI-Powered Digital Twin for B787-9 Landing Gear Health Management

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2024-0076A8?style=flat-square&logo=mathworks)](https://mathworks.com)
[![FlightGear](https://img.shields.io/badge/FlightGear-2024.1-4CAF50?style=flat-square)](https://flightgear.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![ATA](https://img.shields.io/badge/ATA-32%20Landing%20Gear-blue?style=flat-square)]()
[![CS](https://img.shields.io/badge/CS--25.473-Compliant-green?style=flat-square)]()

*A three-layer intelligent MRO system — ODE physics synthesis, CNN-LSTM fault classification, and live FlightGear telemetry integration.*

</div>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AEROTWIN PIPELINE                            │
│                                                                     │
│   FlightGear B787-8                                                 │
│   Simulation (UDP)                                                  │
│        │                                                            │
│        ▼                                                            │
│   mro_dashboard.py  ──►  ODE Physics Engine  ──►  api.py           │
│   (Live Bridge +           odeLanding787()        (Flask)           │
│    Tkinter GUI)            scipy RK45             CNN-LSTM          │
│        │                                          3-head model      │
│        ▼                                               │            │
│   MRO Diagnosis Window  ◄──────────────────────────────            │
│   Class | Severity | RUL | ATA Chapter | Explanation               │
└─────────────────────────────────────────────────────────────────────┘

Layer 1 — Digital Twin:    Simscape multi-body model (validated)
Layer 2 — ML Classifier:   CNN-LSTM (6 signals × 100 timesteps)
Layer 3 — Explanation:     AMM lookup + (LLM planned — see Limitations)
```

---

## Dataset

| Parameter | Value |
|-----------|-------|
| Samples | 18,000 |
| Fault classes | 12 (Classes 0–10, 12) |
| Signal channels | 6 |
| Timesteps per signal | 100 |
| Total feature columns | 632 (32 scalar + 600 signal) |
| Physics engine | Pure ODE (scipy RK45) |
| Generation time | ~8 minutes |
| Certification ref | CS-25.473 / FAR 25.473 |

### Fault Classes

| ID | Class | Samples | ATA |
|----|-------|---------|-----|
| 0 | Normal | 3,500 | 32-00 |
| 1 | N2 Leak | 2,200 | 32-31 |
| 2 | Worn Seal | 2,200 | 32-32 |
| 3 | Early Degradation | 1,600 | 32-00 |
| 4 | Thermal Degradation | 1,400 | 32-61 |
| 5 | Brake Fade | 1,100 | 32-42 |
| 6 | Tire Burst | 900 | 32-41 |
| 7 | Corrosion | 1,100 | 32-31 |
| 8 | Hard Landing [CS-25.473] | 900 | 32-00 |
| 9 | Combined N2+Seal | 800 | 32-31/32 |
| 10 | Impending AOG | 700 | 32-00 |
| 12 | Bogie Pitch [B787] | 1,600 | 32-30 |

---

## Model Architecture

```
Input: (batch, 6, 100)
    │
    ▼
CNN Block
  Conv1d(6→32, k=7) → BN → ReLU → Dropout(0.2)
  Conv1d(32→64, k=5) → BN → ReLU → MaxPool → Dropout(0.2)
  Conv1d(64→128, k=3) → BN → ReLU → MaxPool → Dropout(0.2)
    │
    ▼ (batch, 128, 25)
Bidirectional LSTM (128 hidden, 2 layers, dropout=0.3)
    │
    ▼ (batch, 256) — last timestep
Shared Trunk: Linear(256→256) → BN → ReLU → Linear(256→128) → ReLU
    │
    ├──► head_class:    Linear(128→12)  → Softmax  → Fault class
    ├──► head_severity: Linear(128→4)   → Softmax  → Severity 1–4
    └──► head_rul:      Linear(128→32) → ReLU → Linear(32→1) → Sigmoid → RUL%
```

**Multi-task loss:** `L = 1.0 × L_class + 0.3 × L_severity + 0.2 × L_rul`

---

## Project Structure

```
B787_AeroTwin/
│
├── data/
│   ├── AEROTWIN_B787_V3p0_Dataset.csv        # 18,000 × 632 training dataset
│   └── train_val_split_B787_V3p0.mat         # Pre-defined 70/15/15 split
│
├── models/
│   ├── best_model.pt                          # Best CNN-LSTM checkpoint
│   ├── final_model.pt                         # Final epoch weights
│   ├── signal_scalers.pkl                     # Fitted StandardScalers (train only)
│   └── classification_report.txt             # Per-class F1, precision, recall
│
├── scripts/
│   ├── AEROTWIN_B787_V3p0_DataFactory.m      # MATLAB ODE data generation
│   ├── train_cnn_lstm.py                      # CNN-LSTM training script
│   ├── api.py                                 # Flask inference server
│   ├── mro_dashboard.py                       # Live MRO GUI (replaces live_bridge)
│   ├── udp_sniffer.py                         # UDP stream diagnostic tool
│   └── check.py                               # Dataset validation
│
├── protocol/
│   └── output.xml                             # FlightGear generic UDP protocol
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_demo.ipynb
│
├── environment.yml                            # Conda environment spec
├── requirements.txt                           # pip requirements
└── README.md
```

---

## Quick Start

### 1. Environment

```bash
conda create -n aerospace python=3.10
conda activate aerospace
pip install torch==2.5.1 torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib h5py scipy
pip install flask flask-cors requests anthropic
```

### 2. Generate Dataset (MATLAB)

```matlab
% Run in MATLAB with aerospace environment
run('scripts/AEROTWIN_B787_V3p0_DataFactory.m')
% Output: AEROTWIN_B787_V3p0_Dataset.csv (~8 min)
```

### 3. Train Model

```bash
conda activate aerospace
python scripts/train_cnn_lstm.py
# Output: models/best_model.pt
```

### 4. Live MRO Pipeline

```bash
# Terminal 1 — inference server
python scripts/api.py

# Terminal 2 — live dashboard
python scripts/mro_dashboard.py

# FlightGear (separate)
"C:\Program Files\FlightGear 2024.1\bin\fgfs.exe" \
  --aircraft=787-8 \
  --generic="socket,out,30,localhost,16662,udp,output"
```

Copy `protocol/output.xml` to your FlightGear data Protocol folder first.

---

## Physics Engine

The V3.0 ODE engine implements the B787 oleo-pneumatic strut dynamics:

```
m·ẍ = F_contact(x) - K_eff(x)·x - B·ẋ - F_seal(ẋ) - F_bogie - m·g

where:
  F_contact  = K_c·x^1.5 + B_c·v·x          (Hertz contact)
  K_eff      = K·(1/(1-x/L))^n / R²          (polytropic spring, bogie-corrected)
  F_seal     = Stribeck curve (health-parameterised)
  F_bogie    = geometric correction for semi-levered bogie
```

Validated against Simscape V1.0 and V2.0 multi-body model.
~300× faster than Simscape. No license dependency.

---

## Known Limitations

| Issue | Impact | Status |
|-------|--------|--------|
| RUL head trained on zero-damage data | RUL predictions unreliable | Future work |
| Layer 3 LLM not connected to live pipeline | Explanations are rule-based (AMM CSV) | Future work |
| Simscape Side Stay singularity at retract | V1/V2 limited to touchdown phase only | By design (ODE adopted) |
| FlightGear 787-8 model — missing 3D objects | Visual warnings in FG log (non-critical) | Aircraft model issue |
| No real sensor data validation | Physics is synthetic ODE only | Future work |

---

## Certification Reference

| Standard | Application |
|----------|-------------|
| CS-25.473 / FAR 25.473 | Hard landing limit load threshold (2.0g, 3.05 m/s) |
| ATA 32-00 | Landing Gear — General |
| ATA 32-30 | Extension and Retraction |
| ATA 32-31 | Main Gear and Doors |
| ATA 32-32 | Nose Gear and Doors |
| ATA 32-41 | Wheels and Brakes |
| ATA 32-42 | Brake System |
| ATA 32-61 | Position and Warning |
| AMM 32-00-00-200-801 | Hard Landing Inspection Task |

---

## Results

> ⚠️ Model training results to be updated after full training run completes.

| Metric | Value |
|--------|-------|
| Test Accuracy | TBD |
| Macro F1 | TBD |
| RUL MAE | ⚠️ Unreliable (see Limitations) |
| Dataset generation time | 7.9 minutes (18,000 samples) |
| Inference latency | < 500ms per landing event |

---

## Author

**Sani, M. B.**
Early-Career Aerospace Engineer
Specialisation: MRO Optimisation, Aviation Safety, AI in Maintenance

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@misc{aerotwin2026,
  author    = {Sani, M. B.},
  title     = {AEROTWIN: CNN-LSTM Digital Twin for Aircraft Landing Gear Prognostics},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Sm-bello/aerotwin-b787}
}
```
