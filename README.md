# WaferVision AI — Streamlit Community Cloud

WaferVision AI is a two-stage PyTorch system for detecting and classifying
semiconductor wafer-map defects. This package was extracted from the project's
Kaggle notebook and includes the exact CNNv2 architecture, preprocessing,
probability fusion, Grad-CAM logic, trained checkpoints, and validation samples.

## Repository contents

- `streamlit_app.py`: Streamlit interface and inference code
- `requirements.txt`: Python dependencies using the CPU-only PyTorch index
- `models/`: Stage 1 and Stage 2 trained checkpoints
- `samples/`: `.npy` validation examples exported from Kaggle
- `.streamlit/config.toml`: Streamlit theme and server configuration

## Deploy on Streamlit Community Cloud

1. Put this directory in a public GitHub repository.
2. Sign in at <https://share.streamlit.io>.
3. Connect the GitHub account that owns the repository.
4. Create an app and select:
   - Repository: your WaferVision repository
   - Branch: `main`
   - Entry point: `streamlit_app.py`
   - Python: 3.12
5. Deploy the app.

Community Cloud is free but requires a GitHub account and repository access.
Apps without traffic hibernate after 12 hours and wake when visited.

## Local run

```sh
pip install -r requirements.txt
streamlit run streamlit_app.py
```
