import streamlit as st

from streamlit_webrtc import webrtc_streamer,WebRtcMode
from src.video_processor import AirWritingProcessor


def render_tv():
    st.title("Air Writing")
    
    
    ctx = webrtc_streamer(
        key="air-writing",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AirWritingProcessor,
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },

        media_stream_constraints={
            "video": True,
            "audio": False,
        },

        async_processing=True,
    )


    if ctx.video_processor:
        col1, col2, col3 = st.columns(3)
        col_bool=False

        with col1:
            if st.button("Predict"):
                ctx.video_processor.predict_canvas()
                col_bool=True
        
        # if col_bool:
        #     # with col2:
        #     #     if st.button("Delete Previous Word"):
        #     #         ctx.video_processor.delete_previous_word()
        #     with col3:
        #         if st.button("Reset Sentence"):
        #             ctx.video_processor.reset_sentence()

        st.info(ctx.video_processor.instruction_text)

        if ctx.video_processor.current_word:
            st.write(
                f"Predicted Word: "
                f"{ctx.video_processor.current_word}"
            )
            st.write(
                f"Confidence: "
                f"{ctx.video_processor.current_confidence * 100:.2f}%"
            )

        if ctx.video_processor.last_suggested_words:
            st.subheader("Correction Options")
            cols = st.columns(len(ctx.video_processor.last_suggested_words))
            for idx, word in enumerate(ctx.video_processor.last_suggested_words):
                if cols[idx].button(f"{idx + 1}: {word}"):
                    ctx.video_processor.apply_correction(idx)

        if ctx.video_processor.next_word_suggestions:
            st.subheader("Next Word Suggestions")
            cols = st.columns(len(ctx.video_processor.next_word_suggestions))
            for idx, word in enumerate(ctx.video_processor.next_word_suggestions):
                if cols[idx].button(f"{idx + 4}: {word}"):
                    ctx.video_processor.apply_next_word(idx)

        if ctx.video_processor.current_sentence:
            st.subheader("Sentence")
            st.write(" ".join(ctx.video_processor.current_sentence))
