import streamlit as st
from rag_engine import extract_text_from_url, extract_text_from_pdf, chunk_text, explain_text
st.title("AI Article Explainer (RAG)")

st.write("Provide a URL or upload a PDF to get a grounded explanation.")

# ---- Input controls start here ----

url = st.text_input("Article URL")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

load_source = st.button("Load Document")

if load_source:
    if url:
        with st.spinner("Running RAG pipeline (this can take ~20–40 seconds)..."):
            try:
                text = extract_text_from_url(url)
                st.write("Text extracted. Length:", len(text))

                chunks = chunk_text(text)
                st.session_state["chunks"] = chunks
                st.write("Chunks created:", len(chunks))

            except Exception as e:
                st.error("Pipeline failed")
                st.exception(e)

    elif pdf_file:
        with st.spinner("Processing PDF and running RAG pipeline..."):
            try:
                text = extract_text_from_pdf(pdf_file)
                st.write("Text extracted from PDF. Length:", len(text))

                chunks = chunk_text(text)
                st.session_state["chunks"] = chunks
                st.write("Chunks created:", len(chunks))

            except Exception as e:
                st.error("PDF pipeline failed")
                st.exception(e)

    else:
        st.error("Please provide a URL or upload a PDF.")

if "chunks" in st.session_state:
    st.markdown("---")
    user_query = st.text_area(
        "What do you want explained?",
        placeholder="Be specific. Retrieval depends on this."
    )

    run_explanation = st.button("Run Explanation")

    if run_explanation:
        if user_query.strip():
            explanation = explain_text(st.session_state["chunks"], user_query)
            st.success("Explanation generated")
            st.write(explanation)
        else:
            st.error("Please enter a question or topic to explain.")