import streamlit as st
import sys
import json


def load_data():
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            st.sidebar.success(f"Loaded data from {json_path}")
            return data
        except Exception as e:
            st.sidebar.error(f"Failed to load JSON file: {e}")
            return {}
    else:
        st.sidebar.info("No JSON file provided.")
        return {}


def main():
    st.title("Review LLM Responses by Speaker")

    data = load_data()
    if not data:
        st.warning("No data loaded. Please run with a valid JSON path.")
        return

    speaker_names = list(data.keys())
    selected_speaker = st.selectbox("Select a speaker", speaker_names)

    # Initialize session state
    if 'index' not in st.session_state:
        st.session_state.index = 0
    if 'prev_speaker' not in st.session_state:
        st.session_state.prev_speaker = None

    # Reset index if speaker changes
    if st.session_state.prev_speaker != selected_speaker:
        st.session_state.index = 0
        st.session_state.prev_speaker = selected_speaker

    speaker_data = data[selected_speaker]

    def go_prev():
        st.session_state.index = (st.session_state.index - 1) % len(speaker_data)

    def go_next():
        st.session_state.index = (st.session_state.index + 1) % len(speaker_data)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.button("Previous", on_click=go_prev, key="prev_button")

    with col2:
        st.markdown(f"### Prompt {st.session_state.index + 1} of {len(speaker_data)}")

    with col3:
        st.button("Next", on_click=go_next, key="next_button")

    item = speaker_data[st.session_state.index]

    # Prompt
    st.markdown("#### 🔹 Prompt")
    st.text_area("Prompt", item.get("prompt", ""), height=150)

    # Reference Completion
    st.markdown("#### ✅ Reference Completion")
    st.text_area("True Completion", item.get("true_completion", ""), height=100)

    # Model Responses
    st.markdown("#### 🤖 Model Responses")
    responses = item.get("model_responses", [])
    for i, r in enumerate(responses):
        st.text_area(f"Response {i+1}", r, height=100, key=f"response_{selected_speaker}_{i}_{st.session_state.index}")

    # GPT Response
    gpt_response = item.get("gpt_response")
    if gpt_response:
        st.markdown("#### 💬 GPT Response")
        st.text_area("GPT Response", gpt_response, height=80, key=f"gpt_response_{st.session_state.index}")

    # Final Comparison
    final_comparison = item.get("final_comparison", {})
    if final_comparison:
        st.markdown("#### 🏆 Final Comparison")
        st.write(f"**Winner:** Response {final_comparison.get('winner', '')}")
        st.markdown("**Justification:**")
        st.write(final_comparison.get("justification", ""))

    st.markdown("---")


if __name__ == "__main__":
    main()
