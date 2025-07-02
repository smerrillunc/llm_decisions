import streamlit as st
import pickle
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
            st.sidebar.info("Using dummy data instead.")
            return dummy_data
    else:
        st.sidebar.info("No JSON file provided. Using dummy data.")
        return dummy_data

def main():
    st.title("Speaker Question Analysis")

    data = load_data()

    # Initialize session_state variables
    if 'index' not in st.session_state:
        st.session_state.index = 0
    if 'prev_speaker' not in st.session_state:
        st.session_state.prev_speaker = None
    print(data)
    speaker_names = list(data.keys())
    selected_speaker = st.selectbox("Select a speaker", speaker_names)

    # Reset chunk index to 0 when speaker changes
    if st.session_state.prev_speaker != selected_speaker:
        st.session_state.index = 0
        st.session_state.prev_speaker = selected_speaker

    chunks = data[selected_speaker]

    def go_prev():
        st.session_state.index = (st.session_state.index - 1) % len(chunks)

    def go_next():
        st.session_state.index = (st.session_state.index + 1) % len(chunks)

    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.button("Previous", on_click=go_prev, key="prev_button")

    with col2:
        st.markdown(f"**{st.session_state.index + 1} / {len(chunks)}**")

    with col3:
        st.button("Next", on_click=go_next, key="next_button")

    current_chunk = chunks[st.session_state.index]

    st.markdown(f"### Transcript Utterance:")
    st.write(current_chunk.get('chunk', ''))

    st.markdown(f"### GPT Summary:")
    st.write(current_chunk.get('summary', ''))

    st.markdown(f"### Question:")
    st.write(current_chunk.get('question', ''))

    st.markdown(f"### Fine-Tuned Agent Response:")
    st.write(current_chunk.get('response', ''))

    evaluation = current_chunk.get('evaluation', {})
    if evaluation:
        st.markdown(f"### GPT Score:")
        st.write(evaluation.get('score', ''))

        st.markdown(f"### GPT Explanation:")
        st.write(evaluation.get('explanation', ''))
    else:
        st.markdown(f"### GPT Score:")
        st.write("")
        st.markdown(f"### GPT Explanation:")
        st.write("")

if __name__ == "__main__":
    main()
