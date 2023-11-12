import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_pdf_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap=200,
        length_function=len

    )

    chunks = text_splitter.split_text(raw_text)
    return chunks


#uses the open ai embedding model (paid for)
def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return chain


def handle_user_input(user_input):
    response = st.session_state.conversation({'question':user_input})
    st.write(response["answer"])

def main():
    load_dotenv()
    st.title("Multiple Pdf Information Retreival System")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None


    st.header("Multiple Pdf Information Retreival System")
    user_input = st.text_input("Chat with your pdfs here")

    if(user_input):
        handle_user_input(user_input)

    with st.sidebar:
        st.subheader("Document Upload")
        pdf_docs = st.file_uploader("Upload PDFs here",accept_multiple_files=True)

        if st.button("Proceed"):
            with st.spinner("Proccessing files"):
                text = get_pdf_text(pdf_docs)
                chunks = get_pdf_chunks(text)
                vectorstore = get_vector_store(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("done")



if __name__ == "__main__":
    main()