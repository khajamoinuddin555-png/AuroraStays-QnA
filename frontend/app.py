
import os
import requests
import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AuroraStays QnA",
    page_icon="🏨",
    layout="wide"
)

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:5000"
)

ASK_API = f"{BACKEND_URL}/ask"
HEALTH_API = f"{BACKEND_URL}/health"

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: bold;
    color: #2563EB;
}
.subtitle {
    font-size: 16px;
    color: #64748B;
}
.answer-box {
    background-color: #F0F7FF;
    padding: 18px;
    border-radius: 10px;
    border-left: 5px solid #2563EB;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<p class="main-title">🏨 AuroraStays QnA</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Internal Hotel Policy Question Answering System</p>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Backend Health Status
# --------------------------------------------------

try:
    health_response = requests.get(
        HEALTH_API,
        timeout=5
    )

    if health_response.status_code == 200:
        st.success("Backend API is connected.")
    else:
        st.warning("Backend API is not responding correctly.")

except requests.RequestException:
    st.error(
        "Cannot connect to the backend. "
        "Please start the Flask server."
    )

st.divider()

# --------------------------------------------------
# Question Input
# --------------------------------------------------

st.subheader("Ask a Hotel Policy Question")

with st.form("question_form"):
    question = st.text_area(
        "Enter your question:",
        placeholder="Example: What are the quiet hours at AuroraStay hotels?",
        height=100
    )

    submitted = st.form_submit_button(
        "Ask Question",
        type="primary"
    )

# --------------------------------------------------
# Send Question to Flask Backend
# --------------------------------------------------

if submitted:

    if not question.strip():
        st.warning("Please enter a question.")
    else:

        try:
            with st.spinner("Searching hotel policies and generating an answer..."):

                response = requests.post(
                    ASK_API,
                    json={"question": question.strip()},
                    timeout=120
                )

                response.raise_for_status()
                result = response.json()

            if result.get("status") == "success":

                st.subheader("Answer")

                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True
                )

                # --------------------------------------
                # Supporting Sources
                # --------------------------------------

                st.subheader("Supporting Sources")

                for source in result.get("sources", []):
                    st.markdown(
                        f"- **{source['source']}** "
                        f"(Score: {source['hybrid_score']:.4f})"
                    )

                # --------------------------------------
                # Relevant Retrieved Chunks
                # --------------------------------------

                st.subheader("Relevant Retrieved Chunks")

                for index, chunk in enumerate(
                    result.get("relevant_chunks", []),
                    start=1
                ):

                    with st.expander(
                        f"Chunk {index} | {chunk['source']} | "
                        f"Score: {chunk['hybrid_score']:.4f}"
                    ):

                        st.write(
                            f"**Chunk ID:** {chunk['chunk_id']}"
                        )

                        st.write(chunk["text"])

            else:
                st.error(
                    result.get(
                        "message",
                        "Unable to generate an answer."
                    )
                )

        except requests.exceptions.Timeout:
            st.error(
                "The request timed out. Please try again."
            )

        except requests.exceptions.RequestException as error:
            st.error(
                f"Backend connection error: {error}"
            )

        except Exception as error:
            st.error(
                f"Unexpected error: {error}"
            )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AuroraStays QnA | Internal Proof of Concept | "
    "Answers are grounded in retrieved hotel policy documents."
)
