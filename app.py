"""SafeChat - Child Safety Chat Guardian.
Tab 1: paste ONE message -> AI checks for cyberbullying (ML model).
Tab 2: paste a CONVERSATION -> per-message bullying check + grooming red-flag screen.
Nothing is stored or sent anywhere.
"""
import re
import joblib
import streamlit as st
from grooming import screen_conversation, RISK_TEXT, GROOMING_GUIDANCE

# ---------- load model ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    vec = joblib.load('tfidf.pkl')
    return model, vec

model, vec = load_artifacts()
BULLY_THRESHOLD = 0.45  # tuned on held-out test set; favours catching bullying
NOT_BULLYING = 'not_cyberbullying'

# Human-friendly names for what the model outputs (never show raw labels)
HUMAN_NAMES = {
    'not_cyberbullying': ('Not bullying', 'This message looks okay. No bullying detected.'),
    'age': ('Age-based bullying', 'This message bullies someone because of their age.'),
    'gender': ('Gender-based bullying', 'This message bullies someone because of their gender.'),
    'religion': ('Religion-based bullying', 'This message bullies someone because of their religion.'),
    'ethnicity': ('Ethnicity-based bullying', 'This message bullies someone because of their background.'),
}

BULLYING_GUIDANCE = [
    "Do NOT reply to the bully — replies usually make it worse.",
    "Block the person right away.",
    "Take a screenshot — it is your evidence.",
    "Tell a trusted adult: a parent, teacher, or school counsellor.",
    "Remember: it is not your fault. You deserve to feel safe online.",
]

def clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_bullying(text):
    """Returns (human_name, description, is_bullying, confidence)."""
    import numpy as np
    pr = model.predict_proba(vec.transform([clean(text)]))[0]
    classes = list(model.classes_)
    nb = classes.index(NOT_BULLYING)
    bully_mass = float(1 - pr[nb])
    if bully_mass >= BULLY_THRESHOLD:
        others = [i for i in range(len(classes)) if i != nb]
        top = others[int(np.argmax(pr[others]))]
        name, desc = HUMAN_NAMES[classes[top]]
        return name, desc, True, float(pr[top])
    name, desc = HUMAN_NAMES[NOT_BULLYING]
    return name, desc, False, float(pr[nb])

# ---------- page setup + styling ----------
st.set_page_config(page_title='SafeChat', page_icon='🛡️', layout='centered')

st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, #141e46 0%, #2a4d8f 60%, #3b82c4 100%);
    border-radius: 16px;
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 14px rgba(20,30,70,.25);
}
.main-header h1 { margin: 0; font-size: 2.4rem; }
.main-header p { margin: .4rem 0 0; opacity: .85; font-size: 1.05rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    font-weight: 600;
}
div.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    padding: .6rem 1.6rem;
    background: linear-gradient(135deg, #2a4d8f, #3b82c4);
    color: white;
    border: none;
}
div.stButton > button:hover { filter: brightness(1.1); color: white; }
.verdict-card {
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin: .6rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.verdict-card h4 { margin: 0 0 .3rem; }
.verdict-card p { margin: .2rem 0; }
.guide-box {
    background: #f4f7ff;
    border: 1px solid #dbe6ff;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: .8rem;
}
.guide-box h4 { margin-top: 0; color: #2a4d8f; }
.footer { text-align: center; padding: 2rem 0 .5rem; }
.footer a { margin: 0 14px; font-size: 1.9rem; color: #2a4d8f; text-decoration: none; }
.footer a:hover { color: #3b82c4; }
.footer p { color: #8a8a8a; font-size: .85rem; margin-top: .6rem; }
.msg-row { padding: .35rem 0; border-bottom: 1px solid #f0f0f0; }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

def verdict_card(kind, title, desc, conf):
    styles = {
        'danger': ('#fdeeee', '#c0392b', '⚠️'),
        'ok': ('#eafaf1', '#1e8449', '✅'),
        'warn': ('#fef6e7', '#d68910', '⚠️'),
    }
    bg, border, icon = styles[kind]
    st.markdown(
        f"""<div class="verdict-card" style="background:{bg};border-left:6px solid {border};">
        <h4 style="color:{border};">{icon} {title}</h4>
        <p>{desc}</p>
        <small style="color:#666;">Confidence: {conf:.0%}</small></div>""",
        unsafe_allow_html=True)

def guidance_box(title, steps):
    items = ''.join(f'<p style="margin:.3rem 0;"><b>{i}.</b> {s}</p>' for i, s in enumerate(steps, 1))
    st.markdown(f"""<div class="guide-box"><h4>🧭 {title}</h4>{items}</div>""",
                unsafe_allow_html=True)

# ---------- header ----------
st.markdown("""<div class="main-header">
<h1>🛡️ SafeChat</h1>
<p>Child Safety Chat Guardian — bullying detector & predator warning screen</p>
</div>""", unsafe_allow_html=True)
st.write('Check a message for bullying, or a whole conversation for predator warning signs. '
         'Nothing you type is saved or sent anywhere.')

tab1, tab2 = st.tabs(['🔍 Check a message', '💬 Check a conversation'])

with tab1:
    st.markdown('### Is this message bullying?')
    message = st.text_area('Message to check:', height=120, key='msg',
                           placeholder='Paste the message here...', label_visibility='collapsed')
    if st.button('Check message', type='primary'):
        if not message or not message.strip():
            st.warning('Please paste a message first.')
        elif len(clean(message)) < 3:
            st.warning('That message is too short to check.')
        else:
            name, desc, is_bully, conf = predict_bullying(message)
            if is_bully:
                verdict_card('danger', f'Bullying detected: {name}', desc, conf)
                guidance_box('What to do now', BULLYING_GUIDANCE)
            else:
                verdict_card('ok', name, desc, conf)

with tab2:
    st.markdown('### Check a whole conversation')
    st.write('Paste the chat (one message per line). SafeChat checks every message for bullying '
             'AND scans for predator warning signs.')
    convo = st.text_area('Conversation to check:', height=180, key='convo',
                         placeholder='Paste the conversation here, one message per line...',
                         label_visibility='collapsed')
    if st.button('Check conversation', type='primary'):
        if not convo or not convo.strip():
            st.warning('Please paste a conversation first.')
        else:
            lines = [l.strip() for l in convo.split('\n') if l.strip()]
            st.markdown('#### Message-by-message bullying check')
            any_bully = False
            for l in lines:
                if len(clean(l)) < 3:
                    continue
                name, desc, is_bully, conf = predict_bullying(l)
                icon = '🔴' if is_bully else '🟢'
                if is_bully:
                    any_bully = True
                st.markdown(f'<div class="msg-row">{icon} <b>{name}</b> ({conf:.0%}) — <i>{l[:80]}</i></div>',
                            unsafe_allow_html=True)
            if not any_bully:
                st.success('No bullying detected in these messages.')

            st.markdown('---')
            st.markdown('#### 🚨 Predator warning-sign screen')
            risk, flags = screen_conversation(convo)
            title, desc, _ = RISK_TEXT[risk]
            if risk == 'low':
                verdict_card('ok', title, desc, 1.0)
            elif risk == 'medium':
                verdict_card('warn', title, desc, 1.0)
            else:
                verdict_card('danger', title, desc, 1.0)
            for category, explanation, phrase in flags:
                st.markdown(f'- **{category}**: {explanation} _(matched: "{phrase}")_')
            if risk != 'low':
                guidance_box('What to do now', GROOMING_GUIDANCE)

# ---------- simple footer with social icons ----------
st.markdown("""
<div class="footer">
<a href="https://www.linkedin.com/in/raviteja-buddha-7a9279244" target="_blank" title="LinkedIn"><i class="fab fa-linkedin"></i></a>
<a href="https://github.com/Kushi153" target="_blank" title="GitHub"><i class="fab fa-github"></i></a>
<a href="https://wa.me/919703504923" target="_blank" title="WhatsApp"><i class="fab fa-whatsapp"></i></a>
<p>Built by Buddha Ravi Teja</p>
</div>
""", unsafe_allow_html=True)
