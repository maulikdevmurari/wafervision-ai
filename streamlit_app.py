import io
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


st.set_page_config(
    page_title="WaferVision AI",
    page_icon="🔬",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SAMPLE_DIR = BASE_DIR / "samples"

IMAGE_SIZE = 48
DISPLAY_SIZE = 384
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_WAFER_ELEMENTS = 4_000_000

CLASSES = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
]
DEFECT_CLASSES = CLASSES[1:]
SAMPLE_PATHS = {
    label: SAMPLE_DIR / f"{label}_1.npy"
    for label in CLASSES
}

WAFER_COLORS = np.array(
    [
        [68, 1, 84],
        [33, 145, 140],
        [253, 231, 37],
    ],
    dtype=np.uint8,
)

torch.set_num_threads(1)
DEVICE = torch.device("cpu")
INFERENCE_LOCK = threading.Lock()


class WaferCNNv2(nn.Module):
    """CNN architecture used for both stages in the Kaggle notebook."""

    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image: torch.Tensor):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        output = self.model(image)
        class_idx = int(output.argmax(dim=1).item())
        output[:, class_idx].sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def _load_model(num_classes: int, checkpoint_name: str) -> WaferCNNv2:
    checkpoint_path = MODEL_DIR / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path.name}")

    model = WaferCNNv2(num_classes=num_classes).to(DEVICE)
    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading WaferVision models...")
def load_inference_resources():
    binary_model = _load_model(2, "best_binary_wafer_cnn.pth")
    defect_model = _load_model(8, "best_stage2_defect_cnn.pth")
    gradcam = GradCAM(defect_model, defect_model.features[17])
    return binary_model, defect_model, gradcam


def validate_wafer(wafer: np.ndarray) -> np.ndarray:
    wafer = np.asarray(wafer)
    if wafer.ndim != 2:
        raise ValueError("The wafer must be a two-dimensional array.")
    if wafer.size == 0:
        raise ValueError("The wafer array cannot be empty.")
    if wafer.size > MAX_WAFER_ELEMENTS:
        raise ValueError("The wafer array is too large.")
    if not np.isin(wafer, [0, 1, 2]).all():
        raise ValueError("The wafer must contain only 0, 1, and 2.")
    return wafer


def preprocess_wafer(wafer: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        wafer,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.float32) / 2.0


def create_defect_mask(wafer: np.ndarray) -> np.ndarray:
    return (wafer == 2).astype(np.uint8)


def wafer_to_rgb(wafer: np.ndarray) -> np.ndarray:
    return WAFER_COLORS[wafer.astype(np.uint8)]


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 1] = [255, 220, 0]
    return rgb


def enlarge_for_display(image: np.ndarray) -> np.ndarray:
    return cv2.resize(
        np.asarray(image, dtype=np.uint8),
        (DISPLAY_SIZE, DISPLAY_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


def predict_wafer(wafer: np.ndarray) -> dict:
    wafer = validate_wafer(wafer)
    binary_model, defect_model, gradcam = load_inference_resources()

    processed = preprocess_wafer(wafer)
    image_tensor = (
        torch.from_numpy(np.ascontiguousarray(processed))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with INFERENCE_LOCK:
        with torch.no_grad():
            binary_logits = binary_model(image_tensor)
            defect_probability = torch.softmax(binary_logits, dim=1)[0, 1].item()

            defect_logits = defect_model(image_tensor)
            defect_type_probabilities = (
                torch.softmax(defect_logits, dim=1)[0].cpu().numpy()
            )

        combined_probabilities = np.concatenate(
            [
                [1.0 - defect_probability],
                defect_probability * defect_type_probabilities,
            ]
        )
        heatmap, stage2_class_idx = gradcam.generate(image_tensor)

    predicted_idx = int(np.argmax(combined_probabilities))
    wafer_rgb = cv2.applyColorMap(
        (processed * 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    overlay = cv2.addWeighted(wafer_rgb, 0.65, heatmap_color, 0.35, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return {
        "predicted_class": CLASSES[predicted_idx],
        "defect_probability": defect_probability,
        "stage2_prediction": DEFECT_CLASSES[stage2_class_idx],
        "class_probabilities": {
            CLASSES[index]: float(combined_probabilities[index])
            for index in range(len(CLASSES))
        },
        "wafer_image": enlarge_for_display(wafer_to_rgb(wafer)),
        "mask_image": enlarge_for_display(
            mask_to_rgb(create_defect_mask(wafer))
        ),
        "gradcam_image": enlarge_for_display(overlay),
    }


def load_sample(label: str) -> np.ndarray:
    sample_path = SAMPLE_PATHS[label]
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample file is missing for {label}.")
    return np.load(sample_path, allow_pickle=False)


def load_upload(uploaded_file) -> np.ndarray:
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded file must be 10 MB or smaller.")
    try:
        return np.load(io.BytesIO(file_bytes), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("The uploaded file is not a valid NumPy array.") from error


def store_analysis(wafer: np.ndarray, source: str):
    with st.spinner("Analyzing wafer map..."):
        result = predict_wafer(wafer)
    st.session_state.analysis = result
    st.session_state.analysis_source = source


def render_analysis(result: dict, source: str):
    st.divider()
    st.caption(f"Input: {source}")

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Final prediction", result["predicted_class"])
    metric_2.metric(
        "Defect probability",
        f"{result['defect_probability']:.2%}",
    )
    metric_3.metric("Stage 2 preference", result["stage2_prediction"])

    image_1, image_2, image_3 = st.columns(3)
    image_1.image(result["wafer_image"], caption="Original wafer", width="stretch")
    image_2.image(result["mask_image"], caption="Defect pixel mask", width="stretch")
    image_3.image(
        result["gradcam_image"],
        caption="Stage 2 Grad-CAM",
        width="stretch",
    )

    probability_frame = pd.DataFrame(
        {
            "Class": list(result["class_probabilities"].keys()),
            "Probability": list(result["class_probabilities"].values()),
        }
    ).set_index("Class")

    st.subheader("Nine-class prediction scores")
    st.bar_chart(probability_frame, horizontal=True)
    st.dataframe(
        probability_frame.style.format({"Probability": "{:.2%}"}),
        width="stretch",
    )


st.title("🔬 WaferVision AI")
st.subheader("Intelligent Wafer Defect Detection & Classification")
st.write(
    "WaferVision AI uses a two-stage CNN system. Stage 1 estimates whether a "
    "defect is present, Stage 2 classifies its pattern, and probability fusion "
    "produces a final prediction across eight defect types plus Normal."
)

with st.expander("Model information"):
    st.markdown(
        """
        **Reported held-out performance from the training notebook**

        - Accuracy: **97.29%** on 34,493 wafer maps
        - Macro F1: **85.26%** across all nine classes

        Grad-CAM explains Stage 2's preferred defect category. The defect mask
        displays positions already encoded as defective in the wafer map.
        """
    )

sample_tab, upload_tab = st.tabs(["Validation examples", "Upload wafer map"])

with sample_tab:
    selected_label = st.selectbox(
        "Select defect category",
        options=CLASSES,
        index=CLASSES.index("Donut"),
    )
    if st.button("Analyze example", type="primary", use_container_width=True):
        try:
            store_analysis(load_sample(selected_label), selected_label)
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a categorical `.npy` wafer map",
        type=["npy"],
        help="The array must be two-dimensional and contain only 0, 1, and 2.",
    )
    if st.button(
        "Analyze uploaded wafer",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        try:
            store_analysis(load_upload(uploaded_file), uploaded_file.name)
        except ValueError as error:
            st.error(str(error))

if "analysis" not in st.session_state:
    try:
        store_analysis(load_sample("Donut"), "Donut")
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))

if "analysis" in st.session_state:
    render_analysis(
        st.session_state.analysis,
        st.session_state.analysis_source,
    )
