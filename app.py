import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HateSense ML Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

:root {
    --bg:        #080c14;
    --surface:   #0d1526;
    --card:      #111d35;
    --card2:     #0b1326;
    --accent1:   #00f5d4;
    --accent2:   #f72585;
    --accent3:   #7b2fff;
    --accent4:   #ffd60a;
    --text:      #e2eaf5;
    --muted:     #70819e;
    --border:    rgba(0,245,212,0.18);
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Rajdhani', sans-serif !important;
}

.block-container {
    padding-top: 1.4rem !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #050b18 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.hero-banner {
    background: linear-gradient(135deg, #050b18 0%, #0d1a3a 50%, #0a1226 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px 30px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(0,245,212,0.05) 50%, transparent 100%);
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, var(--accent1), #29b6ff, var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    font-family: 'Share Tech Mono', monospace;
    margin-top: 5px;
}
.status-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent1);
    box-shadow: 0 0 14px var(--accent1);
    margin-right: 8px;
    animation: pulse 1.7s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

.metric-card {
    background: linear-gradient(180deg, #101a31 0%, #0d1730 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 22px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent1), var(--accent3));
}
.metric-label {
    color: var(--muted);
    font-size: 13px;
    font-family: 'Share Tech Mono', monospace;
}
.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--accent1);
    margin-top: 6px;
}

.stepper-wrap {
    background: linear-gradient(180deg, #0a1325 0%, #091120 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 22px;
}
.stepper-grid {
    display: grid;
    grid-template-columns: repeat(8, minmax(0, 1fr));
    gap: 10px;
}
.stepper-item {
    text-align: center;
    opacity: .55;
}
.stepper-item.active {
    opacity: 1;
}
.stepper-circle {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    margin: 0 auto 8px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Orbitron', sans-serif;
    font-weight: 800;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.02);
    color: #60708d;
}
.stepper-item.active .stepper-circle {
    background: linear-gradient(135deg, #5a50ff, #4fbfff);
    color: #fff;
    box-shadow: 0 0 22px rgba(79,191,255,.35);
    border: none;
}
.stepper-title {
    font-size: 12px;
    color: var(--muted);
}
.stepper-item.active .stepper-title {
    color: var(--text);
    font-weight: 700;
}

.section-card {
    background: linear-gradient(180deg, #0d1730 0%, #0a1325 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}
.section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent1);
    margin-bottom: 12px;
}
.step-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.28rem;
    font-weight: 900;
    background: linear-gradient(90deg, var(--accent1), var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.step-desc {
    color: var(--muted);
    font-size: 15px;
    margin-bottom: 18px;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent3), var(--accent1)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    padding: 11px 24px !important;
    box-shadow: 0 6px 24px rgba(123,47,255,.28) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 26px rgba(0,245,212,.30) !important;
}

.stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
}

.stDataFrame {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.result-hate {
    background: linear-gradient(135deg, rgba(247,37,133,.18), rgba(247,37,133,.05));
    border: 2px solid var(--accent2);
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, rgba(0,245,212,.14), rgba(0,245,212,.03));
    border: 2px solid var(--accent1);
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
}
.result-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.45rem;
    font-weight: 900;
}
.result-hate .result-label { color: var(--accent2); }
.result-safe .result-label { color: var(--accent1); }

.small-note {
    color: var(--muted);
    font-size: 12px;
    font-family: 'Share Tech Mono', monospace;
}

@media (max-width: 1100px) {
    .stepper-grid {
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
def init_state():
    defaults = {
        "raw_df": None,
        "clean_df": None,
        "text_col": None,
        "target_col": None,
        "problem_type": "Classification",
        "split_ratio": 25,
        "random_state": 42,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "model": None,
        "y_pred": None,
        "metrics": None,
        "encoded_target_map": None,
        "reverse_target_map": None,
        "prepared_df": None,
        "active_step": "1. Data Input",
        "model_choice": "Logistic Regression",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ============================================================
# HELPERS
# ============================================================
def rgba(r, g, b, a):
    return (r/255.0, g/255.0, b/255.0, a)


def infer_text_column(df: pd.DataFrame):
    preferred = ["tweet", "text", "sentence", "content", "comment", "review", "message"]
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for p in preferred:
        if p in lower_map:
            return lower_map[p]
    obj_cols = [c for c in cols if df[c].dtype == "object"]
    if obj_cols:
        return max(obj_cols, key=lambda x: df[x].astype(str).str.len().mean())
    return cols[0] if cols else None


def infer_label_column(df: pd.DataFrame, text_col=None):
    preferred = ["hate_speech", "label", "target", "class", "category", "sentiment", "output"]
    cols = [c for c in df.columns if c != text_col]
    lower_map = {c.lower(): c for c in cols}
    for p in preferred:
        if p in lower_map:
            return lower_map[p]
    low_card = [c for c in cols if df[c].nunique(dropna=True) <= 20]
    if low_card:
        return low_card[0]
    return cols[-1] if cols else None


def clean_text_series(series: pd.Series):
    cleaned = series.astype(str).str.lower()
    cleaned = cleaned.apply(lambda x: re.sub(r"http\S+|www\S+", " ", x))
    cleaned = cleaned.apply(lambda x: re.sub(r"@[A-Za-z0-9_]+", " ", x))
    cleaned = cleaned.apply(lambda x: re.sub(r"#[A-Za-z0-9_]+", lambda m: " " + m.group(0)[1:] + " ", x))
    cleaned = cleaned.apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", " ", x))
    cleaned = cleaned.apply(lambda x: re.sub(r"\b(rt|amp)\b", " ", x))
    cleaned = cleaned.apply(lambda x: re.sub(r"\s+", " ", x).strip())
    return cleaned


def cyber_axes(fig, ax):
    fig.patch.set_facecolor('#080c14')
    ax.set_facecolor('#0d1526')
    ax.tick_params(colors='#9aa8c0')
    ax.xaxis.label.set_color('#9aa8c0')
    ax.yaxis.label.set_color('#9aa8c0')
    ax.title.set_color('#00f5d4')
    for spine in ax.spines.values():
        spine.set_edgecolor(rgba(0, 245, 212, 0.18))
    ax.grid(axis='y', color=rgba(255, 255, 255, 0.05), linestyle='--', linewidth=0.6)
    return fig, ax


def get_word_freq(series, top_n=12):
    tokens = ' '.join(series.dropna().astype(str)).split()
    return Counter(tokens).most_common(top_n)


def safe_metric(y_true, y_pred, metric_func):
    try:
        return metric_func(y_true, y_pred)
    except Exception:
        return 0.0


def plot_bar(labels, values, colors, title, horizontal=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    if horizontal:
        bars = ax.barh(labels, values, color=colors, edgecolor='none', height=0.58)
        for bar, val in zip(bars, values):
            ax.text(max(val - 0.02, 0.02), bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}' if isinstance(val, float) else str(val),
                    ha='right', va='center', color='#080c14', fontweight='bold', fontsize=10)
        ax.set_xlim(0, max(1.05, max(values) * 1.15 if values else 1.0))
    else:
        bars = ax.bar(labels, values, color=colors, edgecolor='none', width=0.55)
        top_pad = max(values) * 0.06 if values else 1
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top_pad,
                    f'{val:.0f}' if float(val).is_integer() else f'{val:.3f}',
                    ha='center', va='bottom', color='#e2eaf5', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    fig, ax = cyber_axes(fig, ax)
    return fig


def draw_confusion_matrix(cm, labels=("Clean", "Hate")):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.imshow(cm, cmap='magma')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color='#e2eaf5')
    ax.set_yticklabels(labels, color='#e2eaf5')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_edgecolor(rgba(0, 245, 212, 0.18))
    fig.patch.set_facecolor('#080c14')
    ax.set_facecolor('#0d1526')
    return fig


def make_stepper(current_step):
    step_names = [
        "1. Data Input", "2. EDA", "3. Cleaning", "4. Feature Selection",
        "5. Split", "6. Model Selection", "7. Training", "8. Metrics & Predict"
    ]
    short_names = [
        "Data Input", "EDA", "Cleaning", "Features",
        "Split", "Model", "Training", "Metrics"
    ]
    current_idx = step_names.index(current_step) if current_step in step_names else 0
    html = ["<div class='stepper-wrap'><div class='stepper-grid'>"]
    for i, title in enumerate(short_names, start=1):
        cls = "stepper-item active" if i - 1 == current_idx else "stepper-item"
        html.append(
            f"<div class='{cls}'>"
            f"<div class='stepper-circle'>{i}</div>"
            f"<div class='stepper-title'>{title}</div>"
            f"</div>"
        )
    html.append("</div></div>")
    st.markdown(''.join(html), unsafe_allow_html=True)


def prepare_demo_data():
    np.random.seed(42)
    hate_tweets = [
        "i hate those people they are disgusting",
        "kill all of them they dont deserve respect",
        "these animals should be expelled from our country",
        "disgusting filth get out of here",
        "they are ruining everything terrible people",
        "go back where you came from worthless",
        "stupid idiots destroying our society",
        "these criminals should be locked up forever",
        "horrible people spreading disease everywhere",
        "they are subhuman creatures pure evil",
    ]
    clean_tweets = [
        "great day at the park with friends",
        "just finished reading an amazing book",
        "the weather is so beautiful today",
        "excited for the weekend plans",
        "had a wonderful dinner with family",
        "love this new song it is so good",
        "helped a neighbor today felt great",
        "team won the game amazing performance",
        "beautiful sunset photos from the beach",
        "trying a new recipe tonight wish me luck",
    ]
    tweets, labels = [], []
    for _ in range(420):
        if np.random.random() < 0.28:
            tweets.append(np.random.choice(hate_tweets) + " " + " ".join(np.random.choice(['really','very','so','extremely','totally'], size=2)))
            labels.append(1)
        else:
            tweets.append(np.random.choice(clean_tweets) + " " + " ".join(np.random.choice(['today','now','here','again','always'], size=2)))
            labels.append(0)
    return pd.DataFrame({"id": range(1, 421), "tweet": tweets, "label": labels})


def reset_pipeline_after_data_change():
    for key in ["clean_df", "X_train", "X_test", "y_train", "y_test", "model", "y_pred", "metrics", "prepared_df"]:
        st.session_state[key] = None


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 22px;'>
        <div style='font-family:Orbitron,sans-serif; font-size:1.5rem; font-weight:900;
                    background:linear-gradient(90deg,#00f5d4,#7b2fff);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            🛡️ HateSense
        </div>
        <div style='color:#70819e; font-family:"Share Tech Mono",monospace; font-size:11px; margin-top:4px;'>
            ML PIPELINE v3.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.session_state["problem_type"] = st.selectbox(
        "🎯 Problem Type",
        ["Classification", "Regression"],
        index=0
    )

    st.markdown("---")
    st.markdown("<div style='color:#70819e; font-size:13px; font-family:Share Tech Mono, monospace;'>PIPELINE STEPS</div>", unsafe_allow_html=True)
    steps = [
        "1. Data Input", "2. EDA", "3. Cleaning", "4. Feature Selection",
        "5. Split", "6. Model Selection", "7. Training", "8. Metrics & Predict"
    ]
    selected_step = st.radio("", steps, index=steps.index(st.session_state.get("active_step", "1. Data Input")), label_visibility="collapsed")
    st.session_state["active_step"] = selected_step

    st.markdown("---")
    if st.session_state.get("model") is not None:
        model_name = st.session_state.get("model_choice", "Model")
        st.markdown(f"""
        <div style='background:rgba(0,245,212,.08); border:1px solid rgba(0,245,212,.22);
                    border-radius:12px; padding:14px; text-align:center;'>
            <div style='color:#00f5d4; font-family:Orbitron,sans-serif; font-size:11px; font-weight:700;'>MODEL READY</div>
            <div style='color:#e2eaf5; font-size:13px; margin-top:5px;'>{model_name}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-note'>Upload one dataset, choose columns, split with slider, then train.</div>", unsafe_allow_html=True)

step = st.session_state["active_step"]

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero-banner">
    <div style='display:flex; align-items:center; gap:12px;'>
        <div class="status-dot"></div>
        <span style='color:#70819e; font-family:"Share Tech Mono",monospace; font-size:13px;'>SYSTEM ONLINE</span>
    </div>
    <div class="hero-title">🛡️ HateSense ML Dashboard</div>
    <div class="hero-sub">Detection of hate speech using ML model</div>
</div>
""", unsafe_allow_html=True)

make_stepper(step)



# ============================================================
# STEP 1 — DATA INPUT
# ============================================================
if step == "1. Data Input":
    st.markdown('<div class="step-header">STEP 01 — DATA INPUT</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-desc">Upload a single CSV file. The pipeline will later split the data automatically using the split slider.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.35, 0.65])
    with col1:
        st.markdown("**📂 Main Dataset**")
        uploaded = st.file_uploader("Upload file", type=["csv"], key="single_upload")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.session_state["raw_df"] = df
                st.session_state["text_col"] = infer_text_column(df)
                st.session_state["target_col"] = infer_label_column(df, st.session_state["text_col"])
                reset_pipeline_after_data_change()
                st.success(f"✅ Dataset loaded successfully: {len(df)} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Could not read the file: {e}")

        

        if st.button("⚡ Load Demo Data"):
            demo_df = prepare_demo_data()
            st.session_state["raw_df"] = demo_df
            st.session_state["text_col"] = "tweet"
            st.session_state["target_col"] = "label"
            reset_pipeline_after_data_change()
            st.success("✅ Demo dataset loaded.")
            st.rerun()

    # with col2:
    #     st.markdown("""
    #     <div class='section-card'>
    #         <div class='section-title'>✨ What changed</div>
    #         <div style='line-height:1.8;'>
    #             • Single dataset upload<br>
    #             • Automatic split later in pipeline<br>
    #             • Better step UI at top<br>
    #             • Fixed matplotlib RGBA error<br>
    #             • Better model evaluation flow
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    df = st.session_state.get("raw_df")
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL ROWS</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL COLUMNS</div><div class="metric-value">{len(df.columns)}</div></div>', unsafe_allow_html=True)
        with c3:
            txt = st.session_state.get("text_col") or "—"
            st.markdown(f'<div class="metric-card"><div class="metric-label">TEXT COLUMN</div><div class="metric-value" style="font-size:1rem; line-height:1.4;">{txt}</div></div>', unsafe_allow_html=True)
        with c4:
            tgt = st.session_state.get("target_col") or "—"
            st.markdown(f'<div class="metric-card"><div class="metric-label">TARGET COLUMN</div><div class="metric-value" style="font-size:1rem; line-height:1.4; color:#f72585;">{tgt}</div></div>', unsafe_allow_html=True)

        st.markdown("### Preview")
        st.dataframe(df.head(12), use_container_width=True)

# ============================================================
# STEP 2 — EDA
# ============================================================
elif step == "2. EDA":
    st.markdown('<div class="step-header">STEP 02 — EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-desc">Understand class balance, text length, missing values, and frequent words. Chart rendering errors are safely handled.</div>', unsafe_allow_html=True)

    df = st.session_state.get("raw_df")
    if df is None:
        st.warning("⚠️ Please upload a dataset in Step 1 first.")
        st.stop()

    available_cols = list(df.columns)
    default_text = available_cols.index(st.session_state["text_col"]) if st.session_state.get("text_col") in available_cols else 0
    default_target = available_cols.index(st.session_state["target_col"]) if st.session_state.get("target_col") in available_cols else min(1, len(available_cols)-1)

    colA, colB = st.columns(2)
    with colA:
        st.session_state["text_col"] = st.selectbox("Text column", available_cols, index=default_text, key="eda_text_col")
    with colB:
        st.session_state["target_col"] = st.selectbox("Target column", available_cols, index=default_target, key="eda_target_col")

    text_col = st.session_state["text_col"]
    target_col = st.session_state["target_col"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL SAMPLES</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">MISSING VALUES</div><div class="metric-value" style="color:#ffd60a;">{int(df.isna().sum().sum())}</div></div>', unsafe_allow_html=True)
    with c3:
        avg_len = int(df[text_col].astype(str).str.len().mean()) if text_col in df else 0
        st.markdown(f'<div class="metric-card"><div class="metric-label">AVG TEXT LEN</div><div class="metric-value">{avg_len}</div></div>', unsafe_allow_html=True)
    with c4:
        uniq = df[target_col].nunique(dropna=True) if target_col in df else 0
        st.markdown(f'<div class="metric-card"><div class="metric-label">UNIQUE CLASSES</div><div class="metric-value" style="color:#f72585;">{uniq}</div></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Label Distribution**")
        try:
            counts = df[target_col].astype(str).value_counts()
            colors = ['#00f5d4', '#f72585', '#7b2fff', '#ffd60a'][:len(counts)]
            fig = plot_bar(list(counts.index), list(counts.values), colors, 'Label Distribution', horizontal=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Could not render label distribution chart: {e}")

    with col2:
        st.markdown("**📏 Text Length Distribution**")
        try:
            lengths = df[text_col].astype(str).str.len()
            fig, ax = plt.subplots(figsize=(6.8, 4))
            ax.hist(lengths, bins=25, color='#00f5d4', edgecolor='none', alpha=0.88)
            ax.set_xlabel('Text Length')
            ax.set_ylabel('Count')
            ax.set_title('Text Length Distribution', fontsize=13, fontweight='bold')
            fig, ax = cyber_axes(fig, ax)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Could not render text-length chart: {e}")

    st.markdown("---")
    word_cols = st.columns(2)
    try:
        df_temp = df.copy()
        df_temp[text_col] = df_temp[text_col].astype(str)
        labels = list(df_temp[target_col].astype(str).unique())[:2]
        palette = ['#00f5d4', '#f72585']
        for idx, lbl in enumerate(labels):
            subset = df_temp[df_temp[target_col].astype(str) == str(lbl)][text_col]
            freq = get_word_freq(subset, 10)
            if freq:
                words, counts = zip(*freq)
                with word_cols[idx % 2]:
                    fig = plot_bar(list(reversed(words)), list(reversed(counts)), [palette[idx]] * len(words), f'Top Words — Class {lbl}', horizontal=True)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
    except Exception as e:
        st.warning(f"Top-word charts skipped: {e}")

    st.markdown("---")
    st.markdown("**📋 DataFrame Info**")
    buf = io.StringIO()
    df.info(buf=buf)
    st.code(buf.getvalue(), language="text")

    st.markdown("**📊 Missing Values by Column**")
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing": df.isna().sum().values,
        "dtype": [str(t) for t in df.dtypes]
    })
    st.dataframe(missing_df, use_container_width=True)

# ============================================================
# STEP 3 — CLEANING
# ============================================================
elif step == "3. Cleaning":
    st.markdown('<div class="step-header">STEP 03 — TEXT CLEANING</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-desc">Lowercase text, remove URLs, mentions, special characters, and compress spaces.</div>', unsafe_allow_html=True)

    df = st.session_state.get("raw_df")
    if df is None:
        st.warning("⚠️ Please upload a dataset in Step 1 first.")
        st.stop()

    cols = list(df.columns)
    text_idx = cols.index(st.session_state["text_col"]) if st.session_state.get("text_col") in cols else 0
    target_idx = cols.index(st.session_state["target_col"]) if st.session_state.get("target_col") in cols else min(1, len(cols)-1)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["text_col"] = st.selectbox("Select text column", cols, index=text_idx, key="clean_text_col")
    with col2:
        st.session_state["target_col"] = st.selectbox("Select label column", cols, index=target_idx, key="clean_target_col")

    st.markdown("""
    <div class='section-card'>
        <div class='section-title'>🧹 Cleaning Operations</div>
        <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:10px;'>
            <div>✅ Lowercase</div>
            <div>✅ Remove URLs</div>
            <div>✅ Remove mentions</div>
            <div>✅ Remove special characters</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🧼 Run Cleaning Pipeline"):
        try:
            clean_df = df.copy()
            clean_df[st.session_state["text_col"]] = clean_text_series(clean_df[st.session_state["text_col"]])
            clean_df = clean_df.dropna(subset=[st.session_state["text_col"], st.session_state["target_col"]]).reset_index(drop=True)
            st.session_state["clean_df"] = clean_df
            st.session_state["model"] = None
            st.session_state["metrics"] = None
            st.success("✅ Cleaning complete.")
        except Exception as e:
            st.error(f"Cleaning failed: {e}")

    if st.session_state.get("clean_df") is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before Cleaning**")
            st.dataframe(df[[st.session_state["text_col"]]].head(6), use_container_width=True)
        with c2:
            st.markdown("**After Cleaning**")
            st.dataframe(st.session_state["clean_df"][[st.session_state["text_col"]]].head(6), use_container_width=True)

# ============================================================
# STEP 4 — FEATURE SELECTION
# ============================================================
elif step == "4. Feature Selection":
    st.markdown('<div class="step-header">STEP 04 — FEATURE SELECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-desc">Tune the text feature extraction settings. These changes also help reduce unrealistic 100% accuracy.</div>', unsafe_allow_html=True)

    if st.session_state.get("clean_df") is None:
        st.warning("⚠️ Run Cleaning in Step 3 first.")
        st.stop()

    st.markdown("**Primary Text Feature**")
    st.info(f"Using text column: `{st.session_state['text_col']}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        ngram_choice = st.selectbox("N-gram range", ["(1,1)", "(1,2)", "(1,3)"], index=1)
    with col2:
        max_features = st.slider("Max TF-IDF features", 500, 10000, 3000, 500)
    with col3:
        min_df = st.slider("Min document frequency", 1, 10, 2, 1)

    stop_words = st.selectbox("Stop words", ["None", "english"], index=1)
    use_sublinear = st.toggle("Use sublinear TF scaling", value=True)

    ngram_map = {"(1,1)": (1, 1), "(1,2)": (1, 2), "(1,3)": (1, 3)}
    st.session_state["feature_params"] = {
        "ngram_range": ngram_map[ngram_choice],
        "max_features": max_features,
        "min_df": min_df,
        "stop_words": None if stop_words == "None" else "english",
        "sublinear_tf": use_sublinear,
    }

    # st.markdown("""
    # <div class='section-card'>
    #     <div class='section-title'>💡 Accuracy Fix Applied</div>
    #     <div style='line-height:1.8;'>
    #         Earlier 100% accuracy was likely caused by data leakage from balancing before split or from an overly easy split.<br>
    #         This version keeps splitting separate and can use stratified splitting plus better regularization.
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)

# ============================================================
# STEP 5 — SPLIT
# ============================================================
elif step == "5. Split":
    st.markdown('<div class="step-header">STEP 05 — TRAIN / TEST SPLIT</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-desc">Use one slider to split the cleaned dataset into training and testing data automatically.</div>', unsafe_allow_html=True)

    df = st.session_state.get("clean_df")
    if df is None:
        st.warning("⚠️ Run Cleaning in Step 3 first.")
        st.stop()

    test_size = st.slider("Test Size (%)", 10, 40, st.session_state.get("split_ratio", 25), 5)
    random_state = st.number_input("Random State", min_value=0, max_value=9999, value=int(st.session_state.get("random_state", 42)), step=1)
    stratify_enabled = st.toggle("Use stratified split", value=True)

    st.session_state["split_ratio"] = test_size
    st.session_state["random_state"] = int(random_state)

    st.markdown(f"""
    <div class='section-card'>
        <div style='display:flex; gap:20px; align-items:center;'>
            <div style='text-align:center; flex:1;'>
                <div style='font-family:Orbitron,sans-serif; font-size:2rem; font-weight:900; color:#00f5d4;'>{100-test_size}%</div>
                <div style='color:#70819e; font-size:13px; font-family:"Share Tech Mono",monospace;'>TRAINING DATA</div>
            </div>
            <div style='color:#70819e; font-size:2rem;'>⟷</div>
            <div style='text-align:center; flex:1;'>
                <div style='font-family:Orbitron,sans-serif; font-size:2rem; font-weight:900; color:#f72585;'>{test_size}%</div>
                <div style='color:#70819e; font-size:13px; font-family:"Share Tech Mono",monospace;'>TEST DATA</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✂️ Apply Split"):
        try:
            from sklearn.model_selection import train_test_split
            text_col = st.session_state["text_col"]
            target_col = st.session_state["target_col"]

            split_df = df[[text_col, target_col]].dropna().copy()
            y_raw = split_df[target_col]
            if y_raw.dtype == 'object' or str(y_raw.dtype).startswith('category'):
                classes = list(pd.Series(y_raw.astype(str).unique()))
                encoded = {label: idx for idx, label in enumerate(sorted(classes))}
                reverse = {v: k for k, v in encoded.items()}
                y = y_raw.astype(str).map(encoded)
                st.session_state["encoded_target_map"] = encoded
                st.session_state["reverse_target_map"] = reverse
            else:
                y = y_raw.astype(int)
                uniq_vals = sorted(pd.Series(y).unique().tolist())
                st.session_state["reverse_target_map"] = {int(v): str(v) for v in uniq_vals}
                st.session_state["encoded_target_map"] = {str(v): int(v) for v in uniq_vals}

            X = split_df[text_col].astype(str)
            stratify_arg = y if stratify_enabled and len(pd.Series(y).unique()) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size/100,
                random_state=int(random_state),
                stratify=stratify_arg
            )
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.session_state["model"] = None
            st.session_state["metrics"] = None
            st.success(f"✅ Split complete. Train: {len(X_train)} | Test: {len(X_test)}")
        except Exception as e:
            st.error(f"Split failed: {e}")

    if st.session_state.get("X_train") is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">TRAIN SAMPLES</div><div class="metric-value">{len(st.session_state["X_train"])}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">TEST SAMPLES</div><div class="metric-value" style="color:#f72585;">{len(st.session_state["X_test"])}</div></div>', unsafe_allow_html=True)

# ============================================================
# STEP 6 — MODEL SELECTION
# ============================================================
elif step == "6. Model Selection":
    st.markdown('<div class="step-header">STEP 06 — MODEL SELECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-desc">Choose a model and regularization settings. These defaults are selected to reduce overfitting and improve realism.</div>', unsafe_allow_html=True)

    models = [
        "Logistic Regression",
        "Linear SVM",
        "SGD Classifier",
        "Naive Bayes"
    ]
    choice = st.selectbox("Choose Model", models, index=models.index(st.session_state.get("model_choice", "Logistic Regression")))
    st.session_state["model_choice"] = choice

    col1, col2, col3 = st.columns(3)
    if choice == "Logistic Regression":
        with col1:
            C = st.slider("C (inverse regularization)", 0.1, 3.0, 1.0, 0.1)
        with col2:
            class_weight = st.selectbox("Class weight", [None, "balanced"], index=1)
        with col3:
            max_iter = st.slider("Max iterations", 200, 2000, 800, 100)
        st.session_state["model_params"] = {"C": C, "class_weight": class_weight, "max_iter": max_iter}

    elif choice == "Linear SVM":
        with col1:
            C = st.slider("C", 0.1, 3.0, 1.0, 0.1)
        with col2:
            class_weight = st.selectbox("Class weight", [None, "balanced"], index=1, key="svm_class_weight")
        with col3:
            dual = st.selectbox("Dual", [True, False], index=0)
        st.session_state["model_params"] = {"C": C, "class_weight": class_weight, "dual": dual}

    elif choice == "SGD Classifier":
        with col1:
            loss = st.selectbox("Loss", ["hinge", "log_loss", "modified_huber"], index=1)
        with col2:
            alpha = st.number_input("Alpha", value=0.0005, format="%.5f")
        with col3:
            max_iter = st.slider("Max iterations", 200, 3000, 1200, 100)
        penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet"], index=0)
        st.session_state["model_params"] = {"loss": loss, "alpha": alpha, "max_iter": max_iter, "penalty": penalty}

    else:
        with col1:
            alpha = st.number_input("Alpha", value=1.0, format="%.2f")
        with col2:
            fit_prior = st.selectbox("Fit prior", [True, False], index=0)
        with col3:
            st.markdown("<div class='small-note' style='padding-top:28px;'>Best for simple baseline text classification.</div>", unsafe_allow_html=True)
        st.session_state["model_params"] = {"alpha": alpha, "fit_prior": fit_prior}

    st.markdown("""
    <div class='section-card'>
        <div class='section-title'>🏗️ Pipeline Architecture</div>
        <div style='font-family:"Share Tech Mono",monospace; font-size:13px; color:#70819e; line-height:2;'>
            CLEAN TEXT<br>
            &nbsp;&nbsp;↓ TF-IDF Vectorizer<br>
            FEATURE MATRIX<br>
            &nbsp;&nbsp;↓ CLASSIFIER<br>
            PREDICTION + METRICS
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# STEP 7 — TRAINING
# ============================================================
elif step == "7. Training":
    st.markdown('<div class="step-header">STEP 07 — MODEL TRAINING</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-desc">Train the selected model on the training split. This version avoids the earlier train/test leakage problem.</div>', unsafe_allow_html=True)

    if st.session_state.get("X_train") is None:
        st.warning("⚠️ Complete the split in Step 5 first.")
        st.stop()

    if st.button("🚀 Train Model"):
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression, SGDClassifier
            from sklearn.svm import LinearSVC
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            feature_params = st.session_state.get("feature_params", {
                "ngram_range": (1, 2),
                "max_features": 3000,
                "min_df": 2,
                "stop_words": "english",
                "sublinear_tf": True,
            })
            model_choice = st.session_state.get("model_choice", "Logistic Regression")
            model_params = st.session_state.get("model_params", {})

            vectorizer = TfidfVectorizer(
                ngram_range=feature_params["ngram_range"],
                max_features=feature_params["max_features"],
                min_df=feature_params["min_df"],
                stop_words=feature_params["stop_words"],
                sublinear_tf=feature_params["sublinear_tf"]
            )

            if model_choice == "Logistic Regression":
                clf = LogisticRegression(
                    C=model_params.get("C", 1.0),
                    class_weight=model_params.get("class_weight", "balanced"),
                    max_iter=model_params.get("max_iter", 800),
                    random_state=42
                )
            elif model_choice == "Linear SVM":
                clf = LinearSVC(
                    C=model_params.get("C", 1.0),
                    class_weight=model_params.get("class_weight", "balanced"),
                    dual=model_params.get("dual", True),
                    random_state=42
                )
            elif model_choice == "SGD Classifier":
                clf = SGDClassifier(
                    loss=model_params.get("loss", "log_loss"),
                    alpha=model_params.get("alpha", 0.0005),
                    max_iter=model_params.get("max_iter", 1200),
                    penalty=model_params.get("penalty", "l2"),
                    class_weight='balanced',
                    random_state=42
                )
            else:
                clf = MultinomialNB(
                    alpha=model_params.get("alpha", 1.0),
                    fit_prior=model_params.get("fit_prior", True)
                )

            pipeline = Pipeline([
                ('tfidf', vectorizer),
                ('clf', clf)
            ])

            with st.spinner("Training model..."):
                model = pipeline.fit(st.session_state["X_train"], st.session_state["y_train"])
                y_pred = model.predict(st.session_state["X_test"])

            y_test = st.session_state["y_test"]
            metrics = {
                "accuracy": safe_metric(y_test, y_pred, accuracy_score),
                "precision": safe_metric(y_test, y_pred, lambda a, b: precision_score(a, b, average='weighted', zero_division=0)),
                "recall": safe_metric(y_test, y_pred, lambda a, b: recall_score(a, b, average='weighted', zero_division=0)),
                "f1": safe_metric(y_test, y_pred, lambda a, b: f1_score(a, b, average='weighted', zero_division=0)),
            }

            st.session_state["model"] = model
            st.session_state["y_pred"] = y_pred
            st.session_state["metrics"] = metrics
            st.success("✅ Model trained successfully.")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">ACCURACY</div><div class="metric-value">{metrics["accuracy"]:.3f}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">PRECISION</div><div class="metric-value" style="color:#7b2fff;">{metrics["precision"]:.3f}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">RECALL</div><div class="metric-value" style="color:#ffd60a;">{metrics["recall"]:.3f}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">F1 SCORE</div><div class="metric-value" style="color:#f72585;">{metrics["f1"]:.3f}</div></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Training failed: {e}")

    if st.session_state.get("model") is not None:
        st.info(f"Current trained model: **{st.session_state.get('model_choice', 'Model')}**")

# ============================================================
# STEP 8 — METRICS & PREDICT
# ============================================================
elif step == "8. Metrics & Predict":
    st.markdown('<div class="step-header">STEP 08 — METRICS & LIVE PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-desc">Review metrics and test the model on custom text. The earlier EDA and metrics chart error has been fixed.</div>', unsafe_allow_html=True)

    if st.session_state.get("model") is None:
        st.warning("⚠️ Train the model in Step 7 first.")
        st.stop()

    try:
        from sklearn.metrics import confusion_matrix, classification_report
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        metrics = st.session_state.get("metrics", {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">ACCURACY</div><div class="metric-value">{metrics.get("accuracy", 0):.3f}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">PRECISION</div><div class="metric-value" style="color:#7b2fff;">{metrics.get("precision", 0):.3f}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">RECALL</div><div class="metric-value" style="color:#ffd60a;">{metrics.get("recall", 0):.3f}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">F1 SCORE</div><div class="metric-value" style="color:#f72585;">{metrics.get("f1", 0):.3f}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            st.markdown("**🗂️ Confusion Matrix**")
            labels = list(pd.Series(y_test).sort_values().unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            display_labels = [st.session_state.get("reverse_target_map", {}).get(int(lbl), str(lbl)) if str(lbl).isdigit() or isinstance(lbl, (int, np.integer)) else str(lbl) for lbl in labels]
            fig = draw_confusion_matrix(cm, labels=display_labels)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown("**📋 Classification Report**")
            report = classification_report(y_test, y_pred, zero_division=0)
            st.code(report, language="text")

        st.markdown("---")
        st.markdown("**📈 Metrics Overview**")
        fig = plot_bar(
            ['F1 Score', 'Accuracy', 'Precision', 'Recall'],
            [metrics.get("f1", 0), metrics.get("accuracy", 0), metrics.get("precision", 0), metrics.get("recall", 0)],
            ['#00f5d4', '#7b2fff', '#ffd60a', '#f72585'],
            'Performance Metrics',
            horizontal=True
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except Exception as e:
        st.error(f"Could not render metrics section: {e}")

    st.markdown("---")
    st.markdown("#### 🔍 Live Hate Speech Detector")

    tab1, tab2 = st.tabs(["✍️ Single Text", "📋 Batch Predict"])
    with tab1:
        user_text = st.text_area("Enter text to classify:", height=120, placeholder="Type or paste text here...")
        if st.button("🔮 Classify Text"):
            if user_text.strip():
                cleaned = re.sub(r"http\S+|www\S+", " ", user_text.lower())
                cleaned = re.sub(r"@[A-Za-z0-9_]+", " ", cleaned)
                cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                pred = st.session_state["model"].predict([cleaned])[0]
                pred_label = st.session_state.get("reverse_target_map", {}).get(int(pred), str(pred)) if isinstance(pred, (int, np.integer, np.int64)) or str(pred).isdigit() else str(pred)
                if str(pred) == '1' or str(pred_label).lower() in ['1', 'hate', 'hate speech', 'toxic']:
                    st.markdown("""
                    <div class='result-hate'>
                        <div style='font-size:3rem;'>⚠️</div>
                        <div class='result-label'>HATE / TOXIC CONTENT DETECTED</div>
                        <div style='color:#70819e; margin-top:8px;'>The model predicts harmful or hateful language.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='result-safe'>
                        <div style='font-size:3rem;'>✅</div>
                        <div class='result-label'>NON-HATE CONTENT</div>
                        <div style='color:#70819e; margin-top:8px;'>The model predicts non-hateful language.</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"**Cleaned input:** `{cleaned}`")
                st.markdown(f"**Predicted class:** `{pred_label}`")
            else:
                st.warning("Please enter some text first.")

    with tab2:
        batch_input = st.text_area("Enter multiple texts (one per line):", height=180, placeholder="Line 1\nLine 2\nLine 3")
        if st.button("🚀 Classify All"):
            if batch_input.strip():
                lines = [x.strip() for x in batch_input.split('\n') if x.strip()]
                cleaned_lines = []
                for line in lines:
                    cleaned = re.sub(r"http\S+|www\S+", " ", line.lower())
                    cleaned = re.sub(r"@[A-Za-z0-9_]+", " ", cleaned)
                    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()
                    cleaned_lines.append(cleaned)
                preds = st.session_state["model"].predict(cleaned_lines)
                labels = [st.session_state.get("reverse_target_map", {}).get(int(p), str(p)) if isinstance(p, (int, np.integer, np.int64)) or str(p).isdigit() else str(p) for p in preds]
                out = pd.DataFrame({
                    "Original Text": lines,
                    "Cleaned Text": cleaned_lines,
                    "Prediction": labels
                })
                st.dataframe(out, use_container_width=True)

                value_counts = pd.Series(labels).value_counts()
                cols = st.columns(min(3, len(value_counts)))
                for i, (label, count) in enumerate(value_counts.items()):
                    with cols[i % len(cols)]:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{count}</div></div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter at least one line.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:10px 0 4px; font-family:"Share Tech Mono",monospace; font-size:12px; color:#70819e;'>
    🛡️ HateSense ML Dashboard 
</div>
""", unsafe_allow_html=True)
