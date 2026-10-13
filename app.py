"""
Sentiment Analysis - interactive Streamlit app
Run:  streamlit run app.py
"""
import html
import os

import pandas as pd
import streamlit as st

import sentiment_core as sc
from examples import EXAMPLES, next_example

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

MAX_ROWS = 5000
TEXT_COLUMN_GUESSES = ["text", "review", "comment", "tweet", "content", "sentence", "message"]
SAMPLE_CSV = (
    "text\nI love this product, it works perfectly\nThe package arrived on Tuesday\n"
    "This was a terrible experience\nThe hotel is located near the station\n"
    "Awful service and rude staff\n"
).encode("utf-8")
CARD_CLASS = {"Positive": "r-pos", "Negative": "r-neg", "Neutral": "r-neu"}

# Animated background colours. They change with the latest result.
PALETTES = {
    None: ("#c4b5fd", "#93c5fd", "#f9a8d4", "#a5f3fc"),
    "Positive": ("#bbf7d0", "#a7f3d0", "#fde68a", "#99f6e4"),
    "Negative": ("#fecaca", "#fca5a5", "#fdba74", "#f5d0fe"),
    "Neutral": ("#fde68a", "#e5e7eb", "#bfdbfe", "#fed7aa"),
}


def inject_css(label=None):
    c1, c2, c3, c4 = PALETTES[label]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(-45deg, {c1}, {c2}, {c3}, {c4});
            background-size: 400% 400%;
            animation: shift 18s ease infinite;
        }}
        .stApp::before, .stApp::after {{
            content: ""; position: fixed; border-radius: 50%; filter: blur(60px);
            opacity: .55; z-index: 0; pointer-events: none;
            animation: float 22s ease-in-out infinite alternate;
        }}
        .stApp::before {{ width: 380px; height: 380px; background: {c3}; top: 8%; left: 4%; }}
        .stApp::after {{ width: 460px; height: 460px; background: {c2}; bottom: 4%; right: 6%;
                         animation-duration: 28s; animation-direction: alternate-reverse; }}
        @keyframes shift {{ 0% {{background-position: 0% 50%}} 50% {{background-position: 100% 50%}}
                            100% {{background-position: 0% 50%}} }}
        @keyframes float {{ from {{transform: translate(0,0) scale(1)}}
                            to {{transform: translate(80px,-60px) scale(1.2)}} }}
        @media (prefers-reduced-motion: reduce) {{
            .stApp, .stApp::before, .stApp::after {{ animation: none; }} }}

        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: rgba(255,255,255,.72); backdrop-filter: blur(10px); }}
        .block-container, [data-testid="stMainBlockContainer"] {{
            background: rgba(255,255,255,.86); border-radius: 20px; margin-top: 1rem;
            padding: 2rem 2.5rem !important; box-shadow: 0 10px 40px rgba(0,0,0,.08);
            position: relative; z-index: 1; }}

        .hero {{background: linear-gradient(120deg,#7c3aed 0%,#2563eb 100%); padding: 26px 30px;
               border-radius: 16px; color: white; margin-bottom: 18px;}}
        .hero h1 {{color: white; margin: 0; font-size: 2rem;}}
        .hero p {{color: #ede9fe; margin: 6px 0 0 0;}}
        .result {{padding: 18px 24px; border-radius: 14px; font-size: 1.5rem; font-weight: 700;
                 margin: 14px 0;}}
        .r-pos {{background:#dcfce7; color:#166534; border-left:8px solid #16a34a;}}
        .r-neg {{background:#fee2e2; color:#991b1b; border-left:8px solid #dc2626;}}
        .r-neu {{background:#fef9c3; color:#854d0e; border-left:8px solid #ca8a04;}}
        .hl {{padding: 10px 14px; border-radius: 10px; line-height: 2.1; font-size: 1.1rem;
             background: rgba(127,127,127,0.10);}}
        .hl span {{padding: 2px 4px; border-radius: 5px;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model():
    return sc.load_model()


def highlight_html(pairs, text):
    """Words that pushed positive are green, negative are red (stronger = darker)."""
    if not pairs:
        return html.escape(text)
    biggest = max(abs(w) for _, w in pairs) or 1.0
    parts = []
    for word, weight in pairs:
        alpha = min(abs(weight) / biggest, 1.0) * 0.75
        if abs(weight) < 0.02:
            parts.append(html.escape(word))
        else:
            rgb = "22,163,74" if weight > 0 else "220,38,38"
            parts.append(f'<span style="background:rgba({rgb},{alpha:.2f})">{html.escape(word)}</span>')
    return " ".join(parts)


def pick_example(category):
    """Button callback: put the option's next example in the text box (analyzed only on Analyze)."""
    st.session_state["text_input"] = next_example(category, st.session_state["example_pos"])[0]


def classify_frame(model, frame, column):
    """Adds sentiment columns to the rows of `frame` that have text in `column`."""
    out = frame.dropna(subset=[column]).head(MAX_ROWS).copy()
    probs = sc.class_probabilities(model, out[column].astype(str).tolist())
    out["sentiment"] = [sc.label_from_probabilities(p)[0] for p in probs]
    out["confidence %"] = (probs.max(axis=1) * 100).round(1)
    for i, name in enumerate(sc.CLASS_NAMES):
        out[f"{name.lower()} %"] = (probs[:, i] * 100).round(1)
    return out


def main():
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("example_pos", {})
    st.session_state.setdefault("batch_result", None)

    # ---------------- Sidebar ----------------
    metrics = sc.load_metrics()
    st.sidebar.title("📊 Model Information")
    if metrics:
        st.sidebar.metric("Test accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        st.sidebar.metric("F1 score (average of classes)", f"{metrics['f1'] * 100:.1f}%")
    st.sidebar.markdown("**Model**  \nTF-IDF (words and word pairs) + Logistic Regression")
    st.sidebar.markdown("**Classes**  \n😞 Negative, 😐 Neutral, 😊 Positive")
    
    st.sidebar.caption("For educational purposes only.")

    # ---------------- Header ----------------
    st.markdown(
        '<div class="hero"><h1>💬 Sentiment Analysis</h1>'
        "<p>Paste any text and see whether it sounds positive, negative or neutral, "
        "and which words decided it.</p></div>",
        unsafe_allow_html=True,
    )

    if not os.path.exists(sc.MODEL_PATH):
        inject_css()
        st.error("Model not found. Run `python generate_sample_data.py` and then "
                 "`python train_model.py` first.")
        st.stop()
    model = get_model()

    tab_one, tab_batch, tab_hist, tab_model = st.tabs(
        ["✍️ Analyze text", "📁 Batch analysis", "🕘 History", "🧠 About the model"])

    # ---------------- Tab 1: single text ----------------
    with tab_one:
        st.write("Click an option for an example (each click gives a different one), or write your own, then press Analyze:")
        cols = st.columns(len(EXAMPLES))
        for col, label in zip(cols, EXAMPLES):
            col.button(label, key=f"ex_{label}", on_click=pick_example, args=(label,))

        text = st.text_area("Your text", key="text_input", height=130,
                            placeholder="Type or paste a review, tweet, comment...")
        if st.button("🔍 Analyze", type="primary"):
            if not text.strip():
                st.warning("Please enter some text first.")
            else:
                probs = sc.class_probabilities(model, [text])[0]
                st.session_state["result"] = {
                    "text": text, "probs": [float(p) for p in probs],
                    "words": sc.word_contributions(model, text)}
                name, _ = sc.label_from_probabilities(probs)
                st.session_state["history"].append(
                    {"text": text[:120], "sentiment": name,
                     "confidence %": round(float(probs.max()) * 100, 1)})

        res = st.session_state["result"]
        if res:
            name, emoji = sc.label_from_probabilities(res["probs"])
            st.markdown(f'<div class="result {CARD_CLASS[name]}">{emoji} {name}</div>',
                        unsafe_allow_html=True)
            for cls, p in zip(sc.CLASS_NAMES, res["probs"]):
                st.progress(float(p), text=f"{sc.EMOJI[cls]} {cls}: {p * 100:.1f}%")

            st.markdown("**Which words decided it?** (green = pushed positive, red = pushed negative)")
            if res["words"]:
                st.markdown(f'<div class="hl">{highlight_html(res["words"], res["text"])}</div>',
                            unsafe_allow_html=True)
                top = pd.Series({w: v for w, v in res["words"]})
                st.bar_chart(top.reindex(top.abs().sort_values(ascending=False).index).head(8))
            else:
                st.info("Text is too long to highlight individual words.")

    # ---------------- Tab 2: batch ----------------
    with tab_batch:
        mode = st.radio("How do you want to add your texts?",
                        ["📋 Paste text", "📎 Upload a CSV file"], horizontal=True)
        frame, column = pd.DataFrame(), None
        if mode.startswith("📋"):
            pasted = st.text_area("Paste texts (one per line)", height=150, key="batch_text")
            frame = pd.DataFrame({"text": [t.strip() for t in pasted.splitlines() if t.strip()]})
            column = "text"
        else:
            st.caption("Attach a CSV file with a column that holds the text (for example `text` or `review`).")
            st.download_button("⬇️ Download a sample CSV", SAMPLE_CSV,
                               file_name="sample_texts.csv", mime="text/csv")
            upload = st.file_uploader("Attach your CSV file", type="csv")
            if upload is not None:
                try:
                    frame = pd.read_csv(upload)
                except Exception as exc:
                    st.error(f"Could not read that file: {exc}")
                if not frame.empty:
                    guess = next((i for i, c in enumerate(frame.columns)
                                  if str(c).lower() in TEXT_COLUMN_GUESSES), 0)
                    column = st.selectbox("Which column holds the text?", list(frame.columns), index=guess)
                    st.caption(f"{len(frame)} rows found. First rows:")
                    st.dataframe(frame.head(5), hide_index=True)

        ready = column is not None and not frame.empty
        if st.button("▶️ Run batch analysis", disabled=not ready):
            st.session_state["batch_result"] = classify_frame(model, frame, column)
            if len(frame) > MAX_ROWS:
                st.info(f"Only the first {MAX_ROWS} rows were analyzed.")

        out = st.session_state["batch_result"]
        if out is not None:
            counts = out["sentiment"].value_counts().reindex(sc.CLASS_NAMES, fill_value=0)
            for col, cls in zip(st.columns(3), sc.CLASS_NAMES):
                col.metric(f"{sc.EMOJI[cls]} {cls}", int(counts[cls]))
            st.dataframe(out, hide_index=True)
            st.bar_chart(counts)
            st.download_button("⬇️ Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="sentiment_results.csv", mime="text/csv")

    # ---------------- Tab 3: history ----------------
    with tab_hist:
        hist = st.session_state["history"]
        if hist:
            st.dataframe(pd.DataFrame(hist), hide_index=True)
            st.bar_chart(pd.DataFrame(hist)["sentiment"].value_counts())
            if st.button("🗑️ Clear history"):
                st.session_state["history"] = []
        else:
            st.info("Nothing here yet. Analyze some text and it will show up in this list.")

    # ---------------- Tab 4: model ----------------
    with tab_model:
        pos, neg = sc.top_words(model)
        left, right = st.columns(2)
        left.subheader("Most positive words")
        left.bar_chart(pos)
        right.subheader("Most negative words")
        right.bar_chart(neg)
        if metrics:
            st.subheader("Confusion matrix (test set)")
            names = sc.CLASS_NAMES
            st.dataframe(pd.DataFrame(metrics["confusion"],
                                      index=[f"Actually {n.lower()}" for n in names],
                                      columns=[f"Predicted {n.lower()}" for n in names]))

    # The background colour follows the latest result (set last, so it uses this run's result)
    latest = st.session_state["result"]
    inject_css(sc.label_from_probabilities(latest["probs"])[0] if latest else None)


if __name__ == "__main__":
    main()
