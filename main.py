import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import tiktoken

@st.cache_resource
def init_openai_client():
    return OpenAI(api_key=st.secrets["openai_api_key"])

@st.cache_resource
def init_pinecone_client():
    # Use new Pinecone client initialization style (no deprecated init)
    return Pinecone(api_key=st.secrets["pinecone_api_key"])

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

def query_gpt(client, prompt, model="gpt-4o-mini"):
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
    st.markdown(
        """
        Welcome! This app helps you interact with ISO standards documents.
        Select a report and a function from the sidebar, then enter your query or policy text.
        """
    )

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
        - **Policy Presence Check:** Paste a policy text to check if it is addressed in the selected report.
        - **Detailed Chatbot:** Ask detailed questions about the selected report.
        - **Summary & Key Findings:** Generate a summary and key findings of the selected report.
        """
    )

    st.write(f"### Selected Report: {report}")
    st.write(f"### Selected Function: {function}")

    if function == "Summary & Key Findings":
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

    else:
        user_input = st.text_area(
            "Enter your policy text or question here:",
            height=200,
            placeholder="Paste policy text or ask your question..."
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

                if function == "Policy Presence Check":
                    prompt = (
                        "You are an expert assistant. Based on the following document chunks, "
                        "determine if the given policy is present or addressed in the document. "
                        "Provide a detailed explanation with references to the chunks.\n\n"
                        + prompt
                    )
                elif function == "Detailed Chatbot":
                    prompt = (
                        "You are an expert assistant. Use the following document chunks to answer the user's question in detail.\n\n"
                        + prompt
                    )

                answer = query_gpt(openai_client, prompt)
                st.markdown("**Response:**")
                st.write(answer)

    st.markdown("---")
    st.markdown("Developed by NST | Powered by OpenAI GPT-4.1 & Pinecone")

if __name__ == "__main__":
    main()
