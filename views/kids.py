# import streamlit as st
# from streamlit_webrtc import webrtc_streamer,WebRtcMode
# from src.kids_processor import KidsWritingProcessor

# from utils.load_css import load_css
# import streamlit.components.v1 as components


# def render_kids():
#     css = load_css([
#         "assets/kids_styles/style_k.css"
#     ])
#     st.markdown(
#         f"<style>{css}</style>",
#         unsafe_allow_html=True
#     )


#     st.markdown("""
#     <div class="kids-header">
#         <h1>🌈 Kids Learning Zone!</h1>
#         <p>Practice letters and numbers with AI</p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("""
# <div class="practice-card">
#     <h2>🎨 Practice Letters & Digits</h2>
#     <p class="section-title">Pick a Letter</p>
# </div>
# """, unsafe_allow_html=True)


#     letter_cols = st.columns(9)

#     for idx, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
#         if letter_cols[idx % 7].button(letter):
#             st.session_state.target_character = (
#                 letter
#             )

#             st.session_state.mode = (
#                 "emnist_letters"
#             )

   
#     # DIGITS
   
#     st.subheader("Pick a Digit")
#     digit_cols = st.columns(5)
#     for idx in range(10):
#         if digit_cols[idx % 5].button(str(idx)):

#             st.session_state.target_character = (
#                 str(idx)
#             )

#             st.session_state.mode = "mnist"

#     # SHOW TARGET
#     target = st.session_state.get(
#         "target_character"
#     )
#     mode = st.session_state.get(
#         "mode"
#     )

#     if target:

#         st.subheader(
#             f"Practice: {target}"
#         )

#         ctx = webrtc_streamer(

#             key="kids-learning",

#             mode=WebRtcMode.SENDRECV,

#             video_processor_factory=lambda:
#                 KidsWritingProcessor(
#                     target_character=target,
#                     dataset_name=mode,
#                 ),

#             media_stream_constraints={
#                 "video": True,
#                 "audio": False,
#             },

#             async_processing=True,
#         )

#         if ctx.video_processor:

#             col1, col2 = st.columns(2)

#             with col1:

#                 if st.button("✅ Check"):

#                     ctx.video_processor.check_answer()

#             with col2:

#                 if st.button("🧹 Clear"):

#                     ctx.video_processor.clear_canvas()

#             if ctx.video_processor.last_prediction:

#                 st.write(
#                     f"Predicted: "
#                     f"{ctx.video_processor.last_prediction}"
#                 )

#             if ctx.video_processor.result_message:

#                 if "Well done" in (
#                     ctx.video_processor.result_message
#                 ):

#                     st.success(
#                         ctx.video_processor.result_message
#                     )

#                 else:

#                     st.warning(
#                         ctx.video_processor.result_message
#                     )


import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from src.kids_processor import KidsWritingProcessor
from utils.load_css import load_css


def render_kids():

    css = load_css([
        "assets/kids_styles/style_k.css"
    ])

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )

    # =========================
    # HERO
    # =========================

    st.markdown("""
    <div class="kids-hero">
        <h1>🌈 Kids Learning Zone!</h1>
        <p>Practice letters and numbers with AI</p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # MAIN CARD
    # =========================

    with st.container():

        st.markdown("""
        <div class="practice-title">
            🎨 Practice Letters & Digits
        </div>
        """, unsafe_allow_html=True)

        # =========================
        # LETTERS
        # =========================

        st.markdown(
            '<div class="kids-buttons">'
            '<div class="section-title">Pick a Letter</div>',
            unsafe_allow_html=True
        )

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for row in range(3):

            cols = st.columns(9)

            for col in range(9):

                idx = row * 9 + col

                if idx < len(letters):

                    letter = letters[idx]
                    with cols[col]:
                        if st.button(
                            letter,
                            key=f"letter_{letter}",
                            use_container_width=True, 
                        ):

                            st.session_state.target_character = letter
                            st.session_state.mode = "emnist_letters"

        # =========================
        # DIGITS
        # =========================

        st.markdown(
            '<div class="section-title">Pick a Number</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(10)

        for idx in range(10):

            with cols[idx]:

                if st.button(
                    str(idx),
                    key=f"digit_{idx}",
                    use_container_width=True
                ):

                    st.session_state.target_character = str(idx)
                    st.session_state.mode = "mnist"

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # =========================
    # PRACTICE
    # =========================

    target = st.session_state.get("target_character")
    mode = st.session_state.get("mode")

    if target:

        st.markdown(f"""
        <div class="practice-zone">
            <div class="practice-heading">
                Practice: {target}
            </div>
        </div>
        """, unsafe_allow_html=True)

        ctx = webrtc_streamer(

            key="kids-learning",

            mode=WebRtcMode.SENDRECV,

            video_processor_factory=lambda:
                KidsWritingProcessor(
                    target_character=target,
                    dataset_name=mode,
                ),

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            async_processing=True,
        )

        if ctx.video_processor:

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "✅ Check",
                    key="check_btn",
                    use_container_width=True
                ):

                    ctx.video_processor.check_answer()

            with col2:

                if st.button(
                    "🧹 Clear",
                    key="clear_btn",
                    use_container_width=True
                ):

                    ctx.video_processor.clear_canvas()

            if ctx.video_processor.last_prediction:

                st.markdown(f"""
                <div class="prediction-box">
                    Predicted: {ctx.video_processor.last_prediction}
                </div>
                """, unsafe_allow_html=True)

            if ctx.video_processor.result_message:

                if "Well done" in (
                    ctx.video_processor.result_message
                ):

                    st.success(
                        ctx.video_processor.result_message
                    )

                else:

                    st.warning(
                        ctx.video_processor.result_message
                    )