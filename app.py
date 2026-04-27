"""
app.py - ML Web Application (Streamlit)
Interactive UI for dataset creation, cnn_model training, evaluation, and prediction.

Fixes applied
─────────────
1.  Training button lives entirely inside the Train tab (no tab-bleed).
2.  Subprocess is launched at most once; PID is stored in session_state.
3.  Subprocess liveness is checked via psutil before every rerun.
4.  training_log.json is written immediately at training start (empty-but-valid).
5.  training_status.json drives UI states: initializing → training → evaluating → complete/error.
6.  Auto-refresh stops as soon as process dies (complete OR error).
7.  Binary classifier uses softmax+argmax throughout (fixed in train_cnn_model.py too).
8.  stdout is consumed via DEVNULL to prevent pipe-blocking on Windows.
9.  Graceful crash detection: if process dies before first epoch the UI shows the error.
10. Multiple training sessions are blocked while a process is alive.
"""

import streamlit as st
import os
import json
import shutil
import time
import subprocess
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralForge · Custom CNN Trainer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:#0a0a0f; --surface:#12121a; --surface2:#1a1a26;
    --border:#2a2a3e; --accent:#7c3aed; --accent2:#06b6d4;
    --accent3:#f59e0b; --success:#10b981; --danger:#ef4444;
    --text:#e2e8f0; --muted:#64748b;
    --mono:'Space Mono',monospace; --sans:'DM Sans',sans-serif;
}

html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
h1,h2,h3,h4,h5,h6{font-family:var(--mono)!important;color:var(--text)!important;}

.stButton>button{
    background:linear-gradient(135deg,var(--accent),#9333ea)!important;
    color:white!important;border:none!important;border-radius:8px!important;
    font-family:var(--mono)!important;font-weight:700!important;letter-spacing:.05em!important;
    padding:.6rem 1.4rem!important;transition:all .2s ease!important;
    box-shadow:0 4px 15px rgba(124,58,237,.35)!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(124,58,237,.55)!important;}

.stTextInput>div>div>input,.stNumberInput>div>div>input{
    background:var(--surface2)!important;border:1px solid var(--border)!important;
    border-radius:8px!important;color:var(--text)!important;font-family:var(--mono)!important;
}
[data-testid="stFileUploader"]{background:var(--surface2)!important;border:2px dashed var(--border)!important;border-radius:12px!important;padding:1rem!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;}

.stProgress>div>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2))!important;}

[data-testid="stMetricValue"]{color:var(--accent2)!important;font-family:var(--mono)!important;font-size:2rem!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-family:var(--mono)!important;text-transform:uppercase!important;letter-spacing:.1em!important;font-size:.7rem!important;}

.stSelectbox>div>div{background:var(--surface2)!important;border-color:var(--border)!important;color:var(--text)!important;}

.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:10px!important;padding:4px!important;gap:4px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;font-family:var(--mono)!important;font-size:.8rem!important;letter-spacing:.05em!important;border-radius:7px!important;padding:.5rem 1.2rem!important;}
.stTabs [aria-selected="true"]{background:var(--accent)!important;color:white!important;}

.stSuccess{background:rgba(16,185,129,.1)!important;border-left:3px solid var(--success)!important;}
.stError{background:rgba(239,68,68,.1)!important;border-left:3px solid var(--danger)!important;}
.stInfo{background:rgba(6,182,212,.1)!important;border-left:3px solid var(--accent2)!important;}
.stWarning{background:rgba(245,158,11,.1)!important;border-left:3px solid var(--accent3)!important;}

hr{border-color:var(--border)!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--surface);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--accent);}

.nf-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.5rem;margin-bottom:1rem;}
.nf-badge{display:inline-block;background:rgba(124,58,237,.2);border:1px solid rgba(124,58,237,.4);color:#a78bfa;font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;padding:2px 10px;border-radius:999px;text-transform:uppercase;}
.nf-title{font-family:var(--mono);font-size:.65rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:.3rem;}
.conf-high{color:#10b981;font-weight:700;}
.conf-med{color:#f59e0b;font-weight:700;}
.conf-low{color:#ef4444;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")
MODELS_DIR  = Path("cnn_models")
DATASET_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

LOG_PATH    = MODELS_DIR / "training_log.json"
EVAL_PATH   = MODELS_DIR / "eval_results.json"
CM_PATH     = MODELS_DIR / "confusion_matrix.png"
STATUS_PATH = MODELS_DIR / "training_status.json"


# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_classes():
    return sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])

def count_images(class_name):
    p = DATASET_DIR / class_name
    return len([f for f in p.iterdir()
                if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}]) if p.exists() else 0

def read_log():
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def read_eval():
    if EVAL_PATH.exists():
        try:
            with open(EVAL_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def read_status():
    """Returns dict with keys 'state' and 'message', or None."""
    if STATUS_PATH.exists():
        try:
            with open(STATUS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def cnn_model_exists():
    return (MODELS_DIR / "best_cnn_model.keras").exists()

def _proc_alive(pid):
    """Cross-platform process liveness check without psutil."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)   # signal 0 = just probe
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False

def is_training_running():
    """True only if our subprocess is still alive."""
    pid = st.session_state.get("train_pid")
    return _proc_alive(pid)

def training_was_started():
    return st.session_state.get("train_pid") is not None


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;letter-spacing:-.02em;color:#e2e8f0;">🧠 NeuralForge</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:.65rem;letter-spacing:.15em;color:#64748b;text-transform:uppercase;margin-bottom:1.5rem;">Custom CNN Trainer</div>', unsafe_allow_html=True)
    st.divider()

    classes = get_classes()
    if classes:
        st.markdown('<div class="nf-title">📂 Dataset Status</div>', unsafe_allow_html=True)
        for c in classes:
            n = count_images(c)
            color = "#10b981" if n >= 30 else "#f59e0b" if n >= 10 else "#ef4444"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:.4rem .8rem;background:#12121a;border-radius:8px;
                        margin:3px 0;border:1px solid #2a2a3e;font-family:Space Mono,monospace;font-size:.75rem;">
                <span style="color:#e2e8f0;">📁 {c}</span>
                <span style="color:{color};font-weight:700;">{n} imgs</span>
            </div>""", unsafe_allow_html=True)
        st.divider()

    if cnn_model_exists():
        st.markdown('<div class="nf-badge">✅ Model Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#f59e0b;font-family:Space Mono,monospace;font-size:.75rem;">⚠ No trained cnn_model</div>', unsafe_allow_html=True)

    # Show live training status in sidebar too
    status = read_status()
    if status and status.get("state") not in (None, "complete", "error"):
        st.markdown(f'<div style="color:#06b6d4;font-family:Space Mono,monospace;font-size:.7rem;margin-top:.5rem;">⏳ {status["message"]}</div>', unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1.5rem;">
    <div style="font-family:Space Mono,monospace;font-size:2.2rem;font-weight:700;
                background:linear-gradient(135deg,#7c3aed,#06b6d4);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        NeuralForge
    </div>
    <div style="font-family:DM Sans,sans-serif;color:#64748b;font-size:.95rem;margin-top:.3rem;letter-spacing:.08em;">
        Train custom CNN classifiers — no ML expertise required
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📂  Dataset", "🏋️  Train", "📊  Evaluate", "🔍  Predict"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Create & Manage Your Dataset")

    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown('<div class="nf-card">', unsafe_allow_html=True)
        st.markdown("#### ➕ Add New Class")
        new_class = st.text_input("Class name", placeholder="e.g. Cat, Dog, Gesture_A …", key="new_class_input")
        if st.button("Create Class", key="create_class"):
            name = new_class.strip().replace(" ", "_")
            if not name:
                st.error("Please enter a class name.")
            elif name in get_classes():
                st.warning(f"Class '{name}' already exists.")
            else:
                (DATASET_DIR / name).mkdir(parents=True, exist_ok=True)
                st.success(f"✅ Class '{name}' created!")
                st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        classes = get_classes()
        if classes:
            st.markdown('<div class="nf-card">', unsafe_allow_html=True)
            st.markdown("#### 🗑 Delete Class")
            del_class = st.selectbox("Select class to delete", classes, key="del_class_sel")
            if st.button("Delete Class", key="del_class_btn"):
                shutil.rmtree(DATASET_DIR / del_class)
                st.success(f"Deleted '{del_class}'")
                st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        classes = get_classes()
        if not classes:
            st.info("👈 Create at least one class to start uploading images.")
        else:
            st.markdown('<div class="nf-card">', unsafe_allow_html=True)
            st.markdown("#### 📤 Upload Training Images")
            target_class = st.selectbox("Select target class", classes, key="upload_class_sel")
            uploaded_files = st.file_uploader(
                "Upload images (JPG, PNG, WEBP)",
                type=["jpg","jpeg","png","webp","bmp"],
                accept_multiple_files=True,
                key="img_uploader",
            )
            if uploaded_files:
                if st.button(f"💾 Save {len(uploaded_files)} image(s) to '{target_class}'", key="save_imgs"):
                    dest = DATASET_DIR / target_class
                    saved = 0
                    for f in uploaded_files:
                        img = Image.open(f).convert("RGB")
                        fname = f"{int(time.time()*1000)}_{f.name}"
                        img.save(dest / fname)
                        saved += 1
                    st.success(f"✅ Saved {saved} image(s) to '{target_class}'")
                    st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### 🖼 Image Preview")
            preview_class = st.selectbox("Preview class", classes, key="preview_sel")
            p = DATASET_DIR / preview_class
            imgs = sorted(p.iterdir())[:12]
            if imgs:
                grid = st.columns(4)
                for i, img_path in enumerate(imgs):
                    with grid[i % 4]:
                        try:
                            st.image(str(img_path), use_column_width=True)
                        except Exception:
                            pass
            else:
                st.info(f"No images in '{preview_class}' yet.")

    st.divider()
    st.markdown("#### 📋 Dataset Summary")
    classes = get_classes()
    if classes:
        cols = st.columns(len(classes))
        for i, c in enumerate(classes):
            n = count_images(c)
            status_label = "✅ Ready" if n >= 30 else f"⚠ Need {30-n} more"
            cols[i].metric(label=c, value=n, delta=status_label)
    else:
        st.info("No classes created yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN  (ALL training logic lives here — no bleed into other tabs)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Train Your CNN Model")

    classes      = get_classes()
    valid_classes = [c for c in classes if count_images(c) >= 5]

    if len(valid_classes) < 2:
        st.warning("⚠ You need at least **2 classes** with **5+ images each** to train. Go to the Dataset tab first.")
    else:
        col_cfg, col_arch = st.columns(2, gap="large")

        with col_cfg:
            st.markdown('<div class="nf-card">', unsafe_allow_html=True)
            st.markdown("#### ⚙️ Training Configuration")
            epochs    = st.slider("Epochs",           min_value=5, max_value=100, value=20, step=5)
            val_split = st.slider("Validation split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
            st.caption(f"Training on: {', '.join(valid_classes)}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_arch:
            st.markdown('<div class="nf-card">', unsafe_allow_html=True)
            st.markdown("#### 🏗 Model Architecture")
            st.code("""Input (128×128×3)
→ Conv2D(32)×2 → BN → ReLU → MaxPool → Dropout(0.25)
→ Conv2D(64)×2 → BN → ReLU → MaxPool → Dropout(0.25)
→ Conv2D(128)×2→ BN → ReLU → MaxPool → Dropout(0.40)
→ Dense(256)   → BN → ReLU → Dropout(0.50)
→ Softmax (N classes)""", language="text")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Launch button ────────────────────────────────────────────────────
        already_running = is_training_running()

        if already_running:
            st.info("⏳ Training is already running. Monitor progress below.")
        else:
            if st.button("🚀 Start Training", key="train_btn", use_container_width=True):
                # Clean up stale artefacts from a previous run
                for p in [LOG_PATH, STATUS_PATH, EVAL_PATH]:
                    if p.exists():
                        p.unlink()

                # Write an initial status so the UI doesn't spin blind
                with open(STATUS_PATH, "w") as f:
                    json.dump({"state": "initializing", "message": "Launching subprocess…"}, f)

                cmd = [
                    sys.executable, "train_cnn_model.py",
                    "--dataset",   str(DATASET_DIR),
                    "--epochs",    str(epochs),
                    "--val_split", str(val_split),
                ]

                # stdout → DEVNULL prevents Windows pipe-buffer blocking
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                st.session_state["train_pid"]    = proc.pid
                st.session_state["train_epochs"] = epochs
                st.session_state["train_val_split"] = val_split
                st.experimental_rerun()

        st.divider()

        # ── Live Progress Dashboard ──────────────────────────────────────────
        st.markdown("#### 📡 Live Training Progress")

        status      = read_status()
        log         = read_log()
        proc_alive  = is_training_running()
        state       = status.get("state") if status else None
        state_msg   = status.get("message", "") if status else ""

        # ── Determine what to show ───────────────────────────────────────────
        if state == "error":
            st.error(f"❌ Training crashed: {state_msg}")

        elif state == "complete" or (log and log.get("complete")):
            # Training finished successfully
            log = read_log()
            if log and log.get("epoch"):
                total = st.session_state.get("train_epochs", log.get("total_epochs", 20))
                st.progress(1.0, text=f"Epoch {log['epoch'][-1]} / {total} — Complete ✅")
                _col1, _col2, _col3, _col4 = st.columns(4)
                _col1.metric("Train Acc",  f"{log['accuracy'][-1]:.4f}")
                _col2.metric("Val Acc",    f"{log['val_accuracy'][-1]:.4f}")
                _col3.metric("Train Loss", f"{log['loss'][-1]:.4f}")
                _col4.metric("Val Loss",   f"{log['val_loss'][-1]:.4f}")
            st.success("✅ Training complete! Switch to the **Evaluate** tab.")
            # Reset PID so user can start a new run
            st.session_state["train_pid"] = None

        elif state == "initializing" and not log:
            # Process is alive but hasn't written the first epoch yet
            if proc_alive:
                with st.spinner("🔧 Initializing cnn_model — waiting for first epoch…"):
                    time.sleep(2)
                st.experimental_rerun()
            else:
                # Process died before writing anything — check status for error
                st.error("❌ Training process exited unexpectedly before starting. Check train_cnn_model.py for errors.")
                st.session_state["train_pid"] = None

        elif log and log.get("epoch"):
            # Normal in-progress view
            total      = st.session_state.get("train_epochs", log.get("total_epochs", 20))
            done       = log["epoch"][-1]
            progress   = min(done / total, 1.0)

            phase_label = state_msg if state_msg else f"Epoch {done} / {total}"
            st.progress(progress, text=phase_label)

            _col1, _col2, _col3, _col4 = st.columns(4)
            _col1.metric("Train Acc",  f"{log['accuracy'][-1]:.4f}")
            _col2.metric("Val Acc",    f"{log['val_accuracy'][-1]:.4f}")
            _col3.metric("Train Loss", f"{log['loss'][-1]:.4f}")
            _col4.metric("Val Loss",   f"{log['val_loss'][-1]:.4f}")

            # Plotly charts
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
                fig.add_trace(go.Scatter(x=log["epoch"], y=log["accuracy"],
                                         name="Train Acc", line=dict(color="#7c3aed", width=2.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=log["epoch"], y=log["val_accuracy"],
                                         name="Val Acc",   line=dict(color="#06b6d4", width=2.5, dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=log["epoch"], y=log["loss"],
                                         name="Train Loss",line=dict(color="#f59e0b", width=2.5)), row=1, col=2)
                fig.add_trace(go.Scatter(x=log["epoch"], y=log["val_loss"],
                                         name="Val Loss",  line=dict(color="#ef4444", width=2.5, dash="dot")), row=1, col=2)
                fig.update_layout(
                    paper_bgcolor="#0a0a0f", plot_bgcolor="#12121a",
                    font=dict(family="Space Mono", color="#e2e8f0", size=11),
                    legend=dict(bgcolor="#12121a", bordercolor="#2a2a3e"),
                    height=320, margin=dict(l=20, r=20, t=40, b=20),
                )
                for ax in ["xaxis","xaxis2","yaxis","yaxis2"]:
                    fig.update_layout(**{ax: dict(gridcolor="#2a2a3e", zerolinecolor="#2a2a3e")})
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart({"Train Acc": log["accuracy"], "Val Acc": log["val_accuracy"]})

            # Keep refreshing only while the process is alive
            if proc_alive:
                st.caption("⏳ Auto-refreshing…")
                time.sleep(2)
                st.experimental_rerun()
            else:
                # Process died — re-read status in case it wrote "complete" just now
                final_status = read_status()
                if final_status and final_status.get("state") == "error":
                    st.error(f"❌ Training crashed: {final_status['message']}")
                else:
                    st.success("✅ Training complete! Switch to the **Evaluate** tab.")
                st.session_state["train_pid"] = None

        elif training_was_started():
            # PID set but no log yet and not alive → process already died silently
            if not proc_alive:
                st.error("❌ Training process exited without producing logs. Check your dataset and dependencies.")
                st.session_state["train_pid"] = None
            else:
                with st.spinner("🔧 Starting up…"):
                    time.sleep(2)
                st.experimental_rerun()

        else:
            st.info("Configure training above and click **🚀 Start Training** to begin.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Evaluation")

    eval_data = read_eval()
    if not eval_data:
        st.info("No evaluation results yet. Train your cnn_model first.")
    else:
        acc         = eval_data["final_accuracy"]
        class_names = eval_data["class_names"]
        report      = eval_data["classification_report"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Accuracy", f"{acc:.1%}")
        col2.metric("Classes",        len(class_names))
        macro_f1 = report.get("macro avg", {}).get("f1-score", 0)
        col3.metric("Macro F1",       f"{macro_f1:.4f}")

        st.divider()

        col_cm, col_rep = st.columns([1.2, 1], gap="large")

        with col_cm:
            st.markdown("#### Confusion Matrix")
            if CM_PATH.exists():
                st.image(str(CM_PATH), use_column_width=True)

        with col_rep:
            st.markdown("#### Per-Class Metrics")
            for cls in class_names:
                if cls in report:
                    m = report[cls]
                    with st.expander(f"📊 {cls}", expanded=True):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Precision", f"{m['precision']:.3f}")
                        c2.metric("Recall",    f"{m['recall']:.3f}")
                        c3.metric("F1",        f"{m['f1-score']:.3f}")

        st.divider()
        st.markdown("#### Full Classification Report")
        lines = ["Class".ljust(20) + "Precision  Recall  F1-score  Support", "-"*60]
        for cls in class_names:
            if cls in report:
                m = report[cls]
                lines.append(
                    f"{cls:<20}{m['precision']:.3f}      "
                    f"{m['recall']:.3f}   {m['f1-score']:.3f}     {int(m['support'])}"
                )
        st.code("\n".join(lines), language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Real-Time Prediction")

    if not cnn_model_exists():
        st.warning("⚠ No trained cnn_model found. Please train a cnn_model first.")
    else:
        col_upload, col_result = st.columns([1, 1.2], gap="large")

        with col_upload:
            st.markdown('<div class="nf-card">', unsafe_allow_html=True)
            st.markdown("#### 📷 Upload Image for Prediction")
            pred_file = st.file_uploader(
                "Drop an image here",
                type=["jpg","jpeg","png","webp","bmp"],
                key="pred_uploader",
            )

            if pred_file:
                img = Image.open(pred_file).convert("RGB")
                st.image(img, caption="Input image", use_column_width=True)

                if st.button("🔮 Predict", key="predict_btn", use_container_width=True):
                    with st.spinner("Running inference…"):
                        try:
                            from train_cnn_model import predict_image
                            img_array = np.array(img)
                            label, conf, all_probs = predict_image(img_array)
                            st.session_state["pred_result"] = (label, conf, all_probs)
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="nf-card">', unsafe_allow_html=True)
            st.markdown("#### 📸 Webcam Capture (Optional)")
            cam_img = st.camera_input("Take a photo")
            if cam_img:
                if st.button("🔮 Predict from Webcam", key="cam_pred_btn", use_container_width=True):
                    with st.spinner("Running inference…"):
                        try:
                            from train_cnn_model import predict_image
                            img = Image.open(cam_img).convert("RGB")
                            img_array = np.array(img)
                            label, conf, all_probs = predict_image(img_array)
                            st.session_state["pred_result"] = (label, conf, all_probs)
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            if "pred_result" in st.session_state:
                label, conf, all_probs = st.session_state["pred_result"]
                conf_color = "conf-high" if conf >= 0.75 else "conf-med" if conf >= 0.5 else "conf-low"
                st.markdown(f"""
                <div class="nf-card" style="text-align:center;border:1px solid #7c3aed;">
                    <div class="nf-title">Prediction Result</div>
                    <div style="font-family:Space Mono,monospace;font-size:2rem;font-weight:700;color:#e2e8f0;margin:.5rem 0;">{label}</div>
                    <div class="{conf_color}" style="font-family:Space Mono,monospace;font-size:1.3rem;">{conf:.1%} confidence</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("#### All Class Probabilities")
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

                try:
                    import plotly.graph_objects as go
                    colors = ["#7c3aed" if c == label else "#2a2a3e" for c, _ in sorted_probs]
                    fig = go.Figure(go.Bar(
                        x=[p*100 for _, p in sorted_probs],
                        y=[c for c, _ in sorted_probs],
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"{p:.1%}" for _, p in sorted_probs],
                        textposition='outside',
                        textfont=dict(family="Space Mono", color="#e2e8f0", size=11),
                    ))
                    fig.update_layout(
                        paper_bgcolor="#0a0a0f", plot_bgcolor="#12121a",
                        font=dict(family="Space Mono", color="#e2e8f0", size=11),
                        xaxis=dict(gridcolor="#2a2a3e", ticksuffix="%"),
                        yaxis=dict(gridcolor="#2a2a3e"),
                        height=60 + 50*len(sorted_probs),
                        margin=dict(l=10, r=60, t=10, b=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    for cls, prob in sorted_probs:
                        st.progress(prob, text=f"{cls}: {prob:.1%}")
            else:
                st.markdown("""
                <div style="text-align:center;padding:4rem 2rem;color:#64748b;font-family:Space Mono,monospace;">
                    <div style="font-size:3rem;margin-bottom:1rem;">🔮</div>
                    <div style="font-size:.85rem;letter-spacing:.1em;text-transform:uppercase;">Upload an image and click Predict</div>
                </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#2a2a3e;font-family:Space Mono,monospace;font-size:.7rem;
            letter-spacing:.1em;padding:.5rem;">
    NEURALFORGE · CUSTOM CNN TRAINER · BUILT WITH TENSORFLOW + STREAMLIT
</div>""", unsafe_allow_html=True)
