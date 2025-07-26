import streamlit as st
import numpy as np
from pathlib import Path

# File paths
json_path = Path("/playpen-ssd/smerrill/dataset/cleantranscripts/8TdTe--0CUs.npy")
npy_path = Path("/playpen-ssd/smerrill/dataset/transcripts/8TdTe--0CUs.npy")

st.set_page_config(page_title="Transcript Comparison", layout="wide")
st.title("🎙️ Transcript Comparison Viewer")

@st.cache_data
def load_transcripts(json_file, npy_file):
    raw = np.load(npy_file, allow_pickle=True).tolist()
    cleaned = np.load(json_file, allow_pickle=True).tolist()
    return cleaned, raw

cleaned_transcript, raw_transcript = load_transcripts(json_path, npy_path)

# Display metadata
st.markdown(f"**Cleaned Transcript:** {json_path.name}  \n**Raw Transcript:** {npy_path.name}")
st.write(f"Loaded {len(cleaned_transcript)} cleaned turns and {len(raw_transcript)} raw turns.")

# Join all cleaned and raw transcript texts into big strings
cleaned_text = "\n\n".join(f"**{turn['speaker']}**: {turn['text']}" for turn in cleaned_transcript)
raw_text = "\n\n".join(f"**{turn['speaker']}**: {turn['text']}" for turn in raw_transcript)

st.subheader("🔍 Side-by-side Comparison")

# Create two columns
c1, c2 = st.columns(2)

# Inject CSS for scrollable containers with fixed height and dark theme
scroll_container_style = """
<style>
.scrollable-panel {
    height: 500px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #444;
    border-radius: 5px;
    white-space: pre-wrap;
    background-color: #1e1e1e;  /* dark background */
    color: #f0f0f0;             /* light text */
    font-family: monospace;
}
</style>
"""

st.markdown(scroll_container_style, unsafe_allow_html=True)

with c1:
    st.markdown("### Cleaned Transcript")
    st.markdown(f'<div class="scrollable-panel">{cleaned_text}</div>', unsafe_allow_html=True)

with c2:
    st.markdown("### Raw Transcript")
    st.markdown(f'<div class="scrollable-panel">{raw_text}</div>', unsafe_allow_html=True)

if len(cleaned_transcript) != len(raw_transcript):
    st.warning("⚠️ The number of lines in the two transcripts do not match.")
