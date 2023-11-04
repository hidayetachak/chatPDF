import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
import os
import shutil


def get_vectorstore(text):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts([text], embeddings)
    return knowledge_base
def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With any files")
    st.header("💬 ChatPDF")
    st.markdown("### [created By Hidayet](https://etech.org.pk)")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
  
        process = st.button("Process")
        uploaded_files = st.file_uploader("Upload your PDF files here", type=['pdf'], accept_multiple_files=True, key="pdf_uploader")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # analyze PDFs from the "pdfs" directory
        pdfs_directory = "pdfs"
        pdf_files = [os.path.join(pdfs_directory, file) for file in os.listdir(pdfs_directory) if file.endswith(".pdf")]
        pdf_text = get_pdf_text_from_directory(pdf_files)

        # vector stores
        vectorstore = get_vectorstore(pdf_text)

        # conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_user_input(user_question)
    if uploaded_files:
        for file in uploaded_files:
            if file.type == "application/pdf":
                save_pdf_to_directory(file)

# New function to extract text from multiple PDFs
def get_pdf_text_from_directory(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        text += get_pdf_text(pdf_file)
    return text

# Define the get_pdf_text function as before
def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
def save_pdf_to_directory(uploaded_pdf):
    pdfs_directory = "pdfs"
    if not os.path.exists(pdfs_directory):
        os.makedirs(pdfs_directory)
    with open(os.path.join(pdfs_directory, uploaded_pdf.name), "wb") as f:
        f.write(uploaded_pdf.read())
def handle_user_input(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))
        st.write(f"Total Tokens: {cb.total_tokens}" f", Prompt Tokens: {cb.prompt_tokens}" f", Completion Tokens: {cb.completion_tokens}" f", Total Cost (USD): ${cb.total_cost}")
if __name__ == '__main__':
    main()







