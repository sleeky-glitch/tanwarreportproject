import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
from pinecone import Pinecone
import tiktoken

@st.cache_resource
def init_openai_client():
    return OpenAI(api_key=st.secrets["openai_api_key"])

@st.cache_resource
def init_pinecone_client():
    return Pinecone(api_key=st.secrets["pinecone_api_key"])

def extract_text_from_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def embed_text(text: str, client: OpenAI, model="text-embedding-3-small"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def search_pinecone(index, query_embedding, source_filter, top_k=5):
    filter = {"source": {"$eq": source_filter}}
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter,
        namespace="__default__"
    )
    return results.matches

def compose_prompt_for_compliance(policy_text, chunks):
    context_text = "\n\n".join([f"Chunk {i+1}:\n{chunk['metadata']['text']}" for i, chunk in enumerate(chunks)])
    prompt = (
        "You are an expert compliance assistant. Given the following policy text and document chunks from a report, "
        "analyze whether the report complies with the policy. Provide a detailed explanation with reasons, "
        "highlighting which parts of the report comply or do not comply with the policy.\n\n"
        f"Policy Text:\n{policy_text}\n\n"
        f"Report Chunks:\n{context_text}\n\n"
        "Compliance Analysis:"
    )
    return prompt

def compose_prompt(question, chunks):
    context_text = "\n\n".join([f"Chunk {i+1}:\n{chunk['metadata']['text']}" for i, chunk in enumerate(chunks)])
    prompt = (
        "You are an expert assistant helping with ISO standards documents.\n"
        "Use the following extracted document chunks to answer the question in detail.\n\n"
        f"{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt

def query_gpt(client, prompt, model="gpt-4.1"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()

def main():
    st.set_page_config(page_title="ISO Standards Assistant", layout="wide")

    st.title("📄 ISO Standards Document Assistant")

    openai_client = init_openai_client()
    pinecone_client = init_pinecone_client()
    index = pinecone_client.Index(st.secrets["pinecone_index_name"])

    st.sidebar.header("Settings")
    report = st.sidebar.selectbox("Select Report", ["ISO 27001-2022.pdf", "ISO-IEC-42001-2023.pdf"])
    function = st.sidebar.radio("Select Functionality", ["Policy Presence Check", "Detailed Chatbot", "Summary & Key Findings"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Instructions:**")
    st.sidebar.markdown(
        """
        - **Policy Presence Check:** Upload a policy PDF to check if the selected report complies with it.
        - **Detailed Chatbot:** Ask detailed questions about the selected report.
        - **Summary & Key Findings:** Generate a summary and key findings of the selected report.
        """
    )

    st.write(f"### Selected Report: {report}")
    st.write(f"### Selected Function: {function}")

    if function == "Policy Presence Check":
        uploaded_file = st.file_uploader("Upload Policy PDF", type=["pdf"])
        if uploaded_file is not None:
            with st.spinner("Extracting policy text from PDF..."):
                policy_text = extract_text_from_pdf(uploaded_file)
            st.markdown("**Extracted Policy Text Preview:**")
            st.write(policy_text[:1000] + "..." if len(policy_text) > 1000 else policy_text)

            if st.button("Check Compliance"):
                with st.spinner("Embedding policy and searching report..."):
                    policy_embedding = embed_text(policy_text, openai_client)
                    matches = search_pinecone(index, policy_embedding, source_filter=report, top_k=5)

                    if not matches:
                        st.warning("No relevant information found in the selected report.")
                        return

                    for m in matches:
                        if "text" not in m.metadata:
                            m.metadata["text"] = "Text not available."

                    prompt = compose_prompt_for_compliance(policy_text, matches)
                    answer = query_gpt(openai_client, prompt)
                    st.markdown("**Compliance Analysis:**")
                    st.write(answer)

    elif function == "Summary & Key Findings":
        if st.button("Generate Summary & Key Findings"):
            with st.spinner("Generating summary..."):
                query_text = "Provide a detailed summary and key findings of the report."
                query_embedding = embed_text(query_text, openai_client)
                matches = search_pinecone(index, query_embedding, source_filter=report, top_k=5)

                if not matches:
                    st.warning("No relevant information found in the selected report.")
                    return

                prompt = compose_prompt(query_text, matches)
                answer = query_gpt(openai_client, prompt)
                st.markdown("**Summary & Key Findings:**")
                st.write(answer)

    else:  # Detailed Chatbot
        user_input = st.text_area(
            "Enter your question here:",
            height=200,
            placeholder="Ask detailed questions about the selected report..."
        )

        if st.button("Submit") and user_input.strip():
            with st.spinner("Searching and generating response..."):
                query_embedding = embed_text(user_input, openai_client)
                matches = search_pinecone(index, query_embedding, source_filter=report, top_k=5)

                if not matches:
                    st.warning("No relevant information found in the selected report.")
                    return

                for m in matches:
                    if "text" not in m.metadata:
                        m.metadata["text"] = "Text not available."

                prompt = compose_prompt(user_input, matches)

                answer = query_gpt(openai_client, prompt)
                st.markdown("**Response:**")
                st.write(answer)

    st.markdown("---")
    st.markdown("Developed by YourName | Powered by OpenAI GPT-4.1 & Pinecone")

if __name__ == "__main__":
    main()
