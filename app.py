import html
import os
import re
import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_MODEL_DIR = "./final_emotion_distilbert"
HF_FALLBACK_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"

model_source = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else HF_FALLBACK_MODEL
tokenizer = AutoTokenizer.from_pretrained(model_source)
model = AutoModelForSequenceClassification.from_pretrained(model_source, attn_implementation="eager")
model.eval()
model.to(DEVICE)

id2label = model.config.id2label
num_classes = len(id2label)

EMO_THEMES = {
    "joy": {
        "emoji": "🥳",
        "primary": "#10b981",
        "bg_grad": "radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.32), rgba(251, 191, 36, 0.12), #070b14 75%)",
        "anim": "joyBounce 1.1s infinite ease-in-out",
        "tag": "PURE EUPHORIA & POSITIVITY",
        "particles": "✨ 🌟 🎉 ☀️ ✨"
    },
    "sadness": {
        "emoji": "😭",
        "primary": "#38bdf8",
        "bg_grad": "radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.28), rgba(29, 78, 216, 0.15), #050811 75%)",
        "anim": "sadDroop 2.4s infinite ease-in-out",
        "tag": "MELANCHOLY & EMOTIONAL DEPTH",
        "particles": "💧 🌧️ 💙 ☁️ 💧"
    },
    "anger": {
        "emoji": "🤬",
        "primary": "#ef4444",
        "bg_grad": "radial-gradient(circle at 50% 0%, rgba(239, 68, 68, 0.36), rgba(249, 115, 22, 0.16), #0d0507 75%)",
        "anim": "angerShake 0.45s infinite ease-in-out",
        "tag": "HIGH-INTENSITY RAGE & FRUSTRATION",
        "particles": "🔥 💥 ⚡ 💢 🔥"
    },
    "fear": {
        "emoji": "😱",
        "primary": "#a855f7",
        "bg_grad": "radial-gradient(circle at 50% 0%, rgba(168, 85, 247, 0.32), rgba(99, 102, 241, 0.14), #080510 75%)",
        "anim": "fearTremble 0.22s infinite linear",
        "tag": "ANXIETY & THREAT ALERT",
        "particles": "👻 ⚡ 🌑 😰 👻"
    },
    "surprise": {
        "emoji": "🤯",
        "primary": "#f59e0b",
        "bg_grad": "radial-gradient(circle at 50% 0%, rgba(245, 158, 11, 0.35), rgba(236, 72, 153, 0.16), #0d0812 75%)",
        "anim": "surprisePop 1.2s infinite ease-in-out",
        "tag": "SHOCK & UNEXPECTED TWIST",
        "particles": "⚡ 💫 🎊 ❗ ⚡"
    },
    "disgust": {
        "emoji": "🤮",
        "primary": "#84cc16",
        "bg_grad": "radial-gradient(circle at 50% 0%, rgba(132, 204, 22, 0.30), rgba(21, 128, 61, 0.15), #060c06 75%)",
        "anim": "disgustWobble 1.6s infinite ease-in-out",
        "tag": "STRONG AVERSION & REPULSION",
        "particles": "☣️ 🧪 🤢 🦠 ☣️"
    },
    "love": {
        "emoji": "😍",
        "primary": "#ec4899",
        "bg_grad": "radial-gradient(circle at 50% 0%, rgba(236, 72, 153, 0.32), rgba(244, 63, 94, 0.14), #0d050a 75%)",
        "anim": "joyBounce 1.2s infinite ease-in-out",
        "tag": "WARMTH & AFFECTION",
        "particles": "💖 💕 ✨ 🌹 💖"
    }
}

def predict_interactive_mood(text):
    if not text or not text.strip():
        idle_stage = """
        <div class="mood-stage" style="background: radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.18), #070b14 75%); border-color: rgba(56, 189, 248, 0.3);">
            <div class="bouncing-avatar" style="animation: joyBounce 2s infinite ease-in-out;">🎭</div>
            <div class="mood-badge" style="color: #38bdf8; border-color: #38bdf8;">AWAITING YOUR VIBE</div>
            <h2 class="mood-headline">Type anything to trigger the Mood Engine!</h2>
        </div>
        """
        return idle_stage, {}, "<div class='attn-box'>Enter text to see live word-level attention...</div>"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        last_attn = outputs.attentions[-1][0].mean(dim=0)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    dom_emo = id2label[pred_idx].lower()
    dom_conf = float(probs[pred_idx]) * 100
    t = EMO_THEMES.get(dom_emo, EMO_THEMES["joy"])

    conf_dict = {}
    for i in range(num_classes):
        lbl = id2label[i].lower()
        em = EMO_THEMES.get(lbl, {"emoji": "🎭"})["emoji"]
        conf_dict[f"{em} {lbl.upper()}"] = float(probs[i])

    stage_html = f"""
    <div class="mood-stage" style="background: {t['bg_grad']}; border: 2px solid {t['primary']}; box-shadow: 0 0 40px {t['primary']}44;">
        <div class="particle-strip">{t['particles']}</div>
        <div class="bouncing-avatar" style="animation: {t['anim']}; filter: drop-shadow(0 0 24px {t['primary']});">
            {t['emoji']}
        </div>
        <div class="mood-badge" style="color: {t['primary']}; border-color: {t['primary']}; background: {t['primary']}1A;">
            {t['tag']}
        </div>
        <div class="mood-headline" style="color: #ffffff; text-shadow: 0 0 20px {t['primary']}88;">
            {dom_emo.upper()} <span class="conf-pill" style="background: {t['primary']}; color: #050811;">{dom_conf:.1f}%</span>
        </div>
    </div>
    """

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    valid_pairs = [(tok.replace("##", ""), sc) for tok, sc in zip(tokens, last_attn) if tok not in ["[CLS]", "[SEP]", "[PAD]"]]
    scores_arr = np.array([p[1] for p in valid_pairs]) if valid_pairs else np.array([1.0])
    max_s = float(np.max(scores_arr)) if np.max(scores_arr) > 0 else 1.0

    word_spans = []
    for tok, sc in valid_pairs:
        norm_sc = float(sc) / max_s
        alpha = 0.10 + 0.78 * norm_sc
        glow = f"0 0 14px {t['primary']}" if norm_sc > 0.55 else "none"
        scale = "transform: scale(1.08);" if norm_sc > 0.65 else ""
        word_spans.append(
            f"<span class='saliency-chip' style='background: {t['primary']}{int(alpha*255):02x}; "
            f"border: 1px solid {t['primary']}; box-shadow: {glow}; {scale}' "
            f"title='Attention Focus: {norm_sc*100:.1f}%'>{html.escape(tok)}</span>"
        )

    attn_html = f"""
    <div class="attn-panel" style="border-color: {t['primary']}66;">
        <div class="attn-bar" style="background: {t['primary']}22; color: {t['primary']};">
            ⚡ LIVE ATTENTION X-RAY — Tokens Driving the "{dom_emo.upper()}" Reaction
        </div>
        <div class="attn-box">{' '.join(word_spans)}</div>
    </div>
    """
    return stage_html, conf_dict, attn_html

interactive_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;900&display=swap');

body, .gradio-container {
    background: #050811 !important;
    color: #f8fafc !important;
    font-family: 'Outfit', sans-serif !important;
}

@keyframes joyBounce {
    0%, 100% { transform: translateY(0) scale(1) rotate(0deg); }
    30% { transform: translateY(-28px) scale(1.18) rotate(-8deg); }
    60% { transform: translateY(-10px) scale(1.08) rotate(8deg); }
}

@keyframes angerShake {
    0%, 100% { transform: translateX(0) scale(1.12) rotate(0deg); }
    25% { transform: translateX(-12px) scale(1.18) rotate(-7deg); }
    75% { transform: translateX(12px) scale(1.18) rotate(7deg); }
}

@keyframes sadDroop {
    0%, 100% { transform: translateY(0) scale(1) rotate(0deg); }
    50% { transform: translateY(14px) scale(0.94) rotate(-5deg); }
}

@keyframes fearTremble {
    0% { transform: translate(0, 0) scale(1.05); }
    25% { transform: translate(-5px, 4px) scale(1.08); }
    50% { transform: translate(5px, -4px) scale(1.05); }
    75% { transform: translate(-4px, -3px) scale(1.08); }
    100% { transform: translate(4px, 3px) scale(1.05); }
}

@keyframes surprisePop {
    0%, 100% { transform: scale(1); }
    35% { transform: scale(1.32) translateY(-16px); }
    65% { transform: scale(0.96); }
}

@keyframes disgustWobble {
    0%, 100% { transform: rotate(0deg) scale(1); }
    30% { transform: rotate(-14deg) skewX(6deg) scale(1.08); }
    70% { transform: rotate(12deg) skewX(-6deg) scale(1.05); }
}

.mood-stage {
    border-radius: 22px;
    padding: 28px 20px;
    text-align: center;
    transition: all 0.5s cubic-bezier(0.22, 1, 0.36, 1);
    position: relative;
    overflow: hidden;
    margin-bottom: 14px;
}

.particle-strip {
    font-size: 1.35rem;
    letter-spacing: 14px;
    opacity: 0.85;
    margin-bottom: 6px;
}

.bouncing-avatar {
    font-size: 5.5rem;
    line-height: 1.15;
    display: inline-block;
    margin: 8px 0 14px 0;
    cursor: pointer;
    user-select: none;
}

.mood-badge {
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 900;
    letter-spacing: 0.14em;
    padding: 5px 14px;
    border-radius: 999px;
    border: 1px solid;
    margin-bottom: 8px;
}

.mood-headline {
    font-size: 2.1rem;
    font-weight: 900;
    margin: 4px 0 0 0;
    letter-spacing: 0.03em;
}

.conf-pill {
    font-size: 1.15rem;
    padding: 3px 14px;
    border-radius: 999px;
    vertical-align: middle;
    margin-left: 8px;
    font-weight: 900;
}

.attn-panel {
    border: 1px solid;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.85);
    overflow: hidden;
    margin-top: 12px;
}

.attn-bar {
    padding: 9px 15px;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.06em;
}

.attn-box {
    padding: 16px;
    line-height: 2.3;
}

.saliency-chip {
    display: inline-block;
    padding: 3px 10px;
    margin: 3px;
    border-radius: 8px;
    color: #ffffff;
    font-weight: 700;
    font-size: 1rem;
    transition: transform 0.2s ease;
}
"""

vibe_prompts = [
    ["I am beyond thrilled and grateful that our AI research paper finally got accepted!"],
    ["Hearing that news broke my heart, I feel so empty and disappointed today."],
    ["Why do people keep lying straight to my face? It makes me absolutely furious!"],
    ["I was trembling with fear walking through that dark empty subway station alone."],
    ["Wait, what?! You seriously won the entire tournament without losing a single match?!"],
    ["That spoiled food smelled so foul and disgusting I nearly threw up."]
]

with gr.Blocks(css=interactive_css, theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"), title="VibePulse AI — Interactive Emotion Studio") as demo:
    with gr.Row():
        with gr.Column(scale=5):
            gr.HTML("""
            <div style="padding: 14px 18px; border-radius: 16px; background: rgba(15,23,42,0.9); border: 1px solid rgba(148,163,184,0.2); margin-bottom: 10px;">
                <h1 style="margin:0; font-size:1.55rem; font-weight:900; background: linear-gradient(90deg, #38bdf8, #10b981, #f59e0b); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    🎭 VibePulse AI — Mood-Reactive Studio
                </h1>
                <p style="margin:4px 0 0 0; color:#94a3b8; font-size:0.86rem;">
                    Watch the entire studio shift colors & animations based on your text's emotion!
                </p>
            </div>
            """)

            txt_in = gr.Textbox(
                lines=4,
                label="✍️ Express Yourself (Type or Pick a Vibe Below)",
                placeholder="Write something happy, furious, shocking, scary, or heartbreaking...",
                value=vibe_prompts[0][0]
            )
            with gr.Row():
                run_btn = gr.Button("🚀 Trigger Mood Engine", variant="primary", scale=3)
                clr_btn = gr.ClearButton([txt_in], value="Reset", scale=1)

            gr.Examples(examples=vibe_prompts, inputs=txt_in, label="⚡ One-Click Mood Triggers")

        with gr.Column(scale=7):
            stage_output = gr.HTML()
            conf_chart = gr.Label(num_top_classes=6, label="📊 Live Emotion Probability Spectrum")
            attn_output = gr.HTML()

    run_btn.click(fn=predict_interactive_mood, inputs=[txt_in], outputs=[stage_output, conf_chart, attn_output])
    txt_in.submit(fn=predict_interactive_mood, inputs=[txt_in], outputs=[stage_output, conf_chart, attn_output])
    demo.load(fn=predict_interactive_mood, inputs=[txt_in], outputs=[stage_output, conf_chart, attn_output])

if __name__ == "__main__":
    demo.launch()