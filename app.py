import streamlit as st
import os
import tempfile
from video_processor import extract_transcript, extract_frames_summary
from rag_engine import VideoRAGEngine

st.set_page_config(page_title="VideoChat", page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#f5f3ef;--white:#ffffff;--border:#e4e0d8;--accent:#c2410c;--accent2:#ea580c;--text:#1c1917;--muted:#78716c;--tag:#fff1ee;--tag-text:#c2410c;}
*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.stApp,.main,.block-container{font-family:'DM Sans',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
[data-testid="stHeader"]{background-color:var(--bg)!important;}
[data-testid="stSidebar"]{background-color:var(--white)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{font-family:'DM Sans',sans-serif!important;}
.stTextInput label,.stSelectbox label,.stFileUploader label{font-size:11px!important;font-weight:500!important;letter-spacing:0.8px!important;text-transform:uppercase!important;color:var(--muted)!important;}
.stTextInput>div>div>input{background-color:var(--bg)!important;border:1px solid var(--border)!important;border-radius:6px!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important;font-size:13px!important;padding:8px 12px!important;}
.stTextInput>div>div>input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(194,65,12,0.1)!important;}
.stButton>button{background-color:var(--accent)!important;color:white!important;border:none!important;border-radius:6px!important;font-family:'DM Sans',sans-serif!important;font-weight:500!important;font-size:13px!important;padding:10px 20px!important;transition:background 0.2s!important;width:100%!important;}
.stButton>button:hover{background-color:var(--accent2)!important;}
#MainMenu,footer{visibility:hidden;}
.user-bubble{background:var(--accent);color:white;border-radius:16px 16px 4px 16px;padding:12px 18px;margin:6px 0 6px 15%;font-size:14px;line-height:1.6;}
.bot-bubble{background:white;color:var(--text);border-radius:4px 16px 16px 16px;padding:14px 18px;margin:6px 15% 6px 0;font-size:14px;line-height:1.7;border:1px solid var(--border);box-shadow:0 1px 4px rgba(0,0,0,0.04);}
.bubble-label{font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:5px;opacity:0.55;}
.ts-tag{display:inline-block;background:var(--tag);color:var(--tag-text);border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;margin:2px 2px 0 0;font-family:monospace;}
.page-header{padding:2rem 0 1.2rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem;}
.page-header h1{font-family:'Fraunces',serif;font-size:1.9rem;font-weight:700;color:var(--text);letter-spacing:-0.5px;}
.page-header p{color:var(--muted);font-size:13px;margin-top:4px;font-weight:300;}
.pill-ready{background:#fff1ee;color:#c2410c;border:1px solid #fed7aa;border-radius:20px;padding:3px 12px;font-size:11px;font-weight:600;display:inline-block;}
.pill-wait{background:#fef9ec;color:#92690e;border:1px solid #fde68a;border-radius:20px;padding:3px 12px;font-size:11px;font-weight:600;display:inline-block;}
.empty-state{text-align:center;padding:4rem 2rem;color:var(--muted);}
.empty-state h3{font-size:15px;font-weight:500;color:var(--text);margin-bottom:6px;}
.empty-state p{font-size:13px;line-height:1.7;}
.sdiv{border:none;border-top:1px solid var(--border);margin:14px 0;}
.video-info{background:white;border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:8px 0;font-size:13px;}
.video-info b{color:var(--accent);}
</style>
""", unsafe_allow_html=True)

# Session state
for k,v in [("messages",[]),("rag_engine",None),("video_loaded",False),("video_name","")]:
    if k not in st.session_state: st.session_state[k] = v

# Sidebar
with st.sidebar:
    st.markdown("<div style='padding:8px 0 4px'><span style='font-family:Fraunces,serif;font-size:1.2rem;font-weight:700'>VideoChat</span></div>", unsafe_allow_html=True)
    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;color:#78716c;margin-bottom:6px'>Configuration</p>", unsafe_allow_html=True)
    gemini_key = st.text_input("Google Gemini API Key", type="password", placeholder="AIza...")
    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;color:#78716c;margin-bottom:6px'>Upload Video</p>", unsafe_allow_html=True)

    video_source = st.selectbox("Video source", ["Upload a file", "YouTube URL"])

    video_file = None
    youtube_url = ""

    if video_source == "Upload a file":
        video_file = st.file_uploader("Video file", type=["mp4","mov","avi","mkv","webm"])
    else:
        youtube_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")

    st.markdown("<br>", unsafe_allow_html=True)
    process_btn = st.button("Process Video", use_container_width=True)

    if st.session_state.video_loaded:
        st.markdown(f'<div style="margin-top:8px"><span class="pill-ready">Ready — {st.session_state.video_name}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin-top:8px"><span class="pill-wait">No video loaded</span></div>', unsafe_allow_html=True)

    st.markdown("<hr class='sdiv'>", unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("<div style='font-size:11px;color:#aaa;line-height:1.9;margin-top:8px'>1. Enter Gemini API key<br>2. Upload a video or paste YouTube URL<br>3. Click Process Video<br>4. Ask anything about the video</div>", unsafe_allow_html=True)

# Process video
if process_btn:
    if not gemini_key:
        st.error("Please enter your Google Gemini API key.")
    elif video_source == "Upload a file" and not video_file:
        st.error("Please upload a video file.")
    elif video_source == "YouTube URL" and not youtube_url:
        st.error("Please enter a YouTube URL.")
    else:
        with st.spinner("Processing video... this may take a moment."):
            try:
                os.environ["GEMINI_API_KEY"] = gemini_key

                if video_source == "Upload a file":
                    suffix = "." + video_file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(video_file.read())
                        tmp_path = tmp.name
                    transcript = extract_transcript(tmp_path, gemini_key)
                    video_name = video_file.name
                    os.unlink(tmp_path)
                else:
                    from video_processor import get_youtube_transcript
                    transcript = get_youtube_transcript(youtube_url)
                    video_name = youtube_url.split("v=")[-1][:20] + "..."

                engine = VideoRAGEngine(transcript, gemini_key)
                st.session_state.rag_engine  = engine
                st.session_state.video_loaded = True
                st.session_state.video_name   = video_name
                st.session_state.messages     = []
                st.success(f"Video processed! Extracted {len(transcript.split())} words of content.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to process video: {e}")

# Main
st.markdown('<div class="page-header"><h1>VideoChat</h1><p>Ask questions about your video content using AI</p></div>', unsafe_allow_html=True)

if st.session_state.video_loaded:
    engine = st.session_state.rag_engine
    st.markdown(f"""
    <div class="video-info">
        <b>Video loaded:</b> {st.session_state.video_name}<br>
        <b>Content length:</b> ~{len(engine.transcript.split())} words indexed
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.messages:
    if st.session_state.video_loaded:
        st.markdown('<div class="empty-state"><h3>Video processed</h3><p>Ask anything about the video content below.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state"><h3>No video loaded</h3><p>Enter your Gemini API key and upload a video<br>or paste a YouTube URL in the sidebar.</p></div>', unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble"><div class="bubble-label">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            timestamps_html = ""
            if msg.get("timestamps"):
                tags = "".join(f'<span class="ts-tag">{t}</span>' for t in msg["timestamps"][:5])
                timestamps_html = f'<div style="margin-top:10px;border-top:1px solid #e4e0d8;padding-top:8px"><div style="font-size:10px;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;color:#78716c;margin-bottom:5px">Referenced sections</div>{tags}</div>'
            st.markdown(f'<div class="bot-bubble"><div class="bubble-label">Assistant</div>{msg["content"]}{timestamps_html}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("Message", placeholder="Ask about the video...", label_visibility="collapsed", disabled=not st.session_state.video_loaded)
with col2:
    send_btn = st.button("Send", use_container_width=True, disabled=not st.session_state.video_loaded)

if (send_btn or user_input) and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    with st.spinner("Thinking..."):
        try:
            os.environ["GEMINI_API_KEY"] = gemini_key
            answer, timestamps = st.session_state.rag_engine.query(user_input.strip())
            st.session_state.messages.append({"role": "assistant", "content": answer, "timestamps": timestamps})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}", "timestamps": []})
    st.rerun()