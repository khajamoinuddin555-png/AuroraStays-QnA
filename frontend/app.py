
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

ASK_API = f"{BACKEND_URL}/v1/answer_with_relevant_chunks"
HEALTH_API = f"{BACKEND_URL}/health"

# --------------------------------------------------
# SIDEBAR - CONFIGURABLE PARAMETERS
# --------------------------------------------------

st.sidebar.header("⚙️ Parameters")

# Number of chunks to retrieve
k = st.sidebar.slider(
    "K (chunks to retrieve)",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

# LLM temperature
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.0,
    step=0.1
)

# Maximum response tokens
max_tokens = st.sidebar.slider(
    "Max tokens",
    min_value=64,
    max_value=2048,
    value=512,
    step=64
)



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

    response_mode = st.radio(
        "Response Mode:",
        [
            "Answer only",
            "Answer + Relevant Chunks"
        ],
        horizontal=True
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
            with st.spinner(
                "Searching hotel policies and generating an answer..."
            ):
                response = requests.post(
                    ASK_API,
                    json={
                        "query": question.strip(),
                        "k": k,
                        "temperature": temperature,
                        "top_p": 1,
                        "max_tokens": max_tokens
                    },
                    timeout=120
                )

                response.raise_for_status()
                result = response.json()

            if result.get("status") == "success":

                # ----------------------------------
                # Display Answer
                # ----------------------------------

                st.subheader("Answer")

                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True
                )

              
                # ----------------------------------
                # Display Supporting Sources
                # ----------------------------------

                if response_mode == "Answer + Relevant Chunks":

                    chunks = result.get("chunks", [])

                    st.subheader("Supporting Sources")

                    if chunks:
                        unique_sources = list(dict.fromkeys(
                            chunk["metadata"]["source"]
                            for chunk in chunks
                        ))

                        for source in unique_sources:
                            st.markdown(f"- **{source}**")

                    else:
                        st.info("No supporting sources found.")

                    # ----------------------------------
                    # Relevant Retrieved Chunks
                    # ----------------------------------

                    st.subheader("Relevant Retrieved Chunks")

                    if chunks:
                        for index, chunk in enumerate(chunks, start=1):

                            metadata = chunk["metadata"]

                            with st.expander(
                                f"Chunk {index} | {metadata['source']} | "
                                f"Score: {metadata['hybrid_score']:.4f}"
                            ):
                                st.write(
                                    f"**Chunk ID:** {metadata['chunk_id']}"
                                )

                                st.write(chunk["text"])

                    else:
                        st.info("No relevant chunks found.")

            else:
                st.error(
                    result.get(
                        "message",
                        "Unable to generate an answer."
                    )
                )

        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again.")

        except requests.exceptions.RequestException as error:
            st.error(f"Backend connection error: {error}")

        except Exception as error:
            st.error(f"Unexpected error: {error}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AuroraStays QnA | Internal Proof of Concept | "
    "Answers are grounded in retrieved hotel policy documents."
)
