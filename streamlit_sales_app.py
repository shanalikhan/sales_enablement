""" streamlit app code """
import streamlit as st
from streamlit_chat import message
from pages import home_page, chatbot, upload
from utils import ChromaManager, LLM

chroma_manager = ChromaManager()
llm = LLM()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

st.set_page_config(layout="wide")

# Define session state for data retrievers if not already defined
if 'casedb_retriever' not in st.session_state or 'callsdb_retriever' not in st.session_state:
    casedb_retriever, callsdb_retriever = chroma_manager.get_retriever()
    st.session_state.casedb_retriever = casedb_retriever
    st.session_state.callsdb_retriever = callsdb_retriever
    st.session_state.message = message

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Create buttons for page selection in the sidebar
home_button = st.sidebar.button("Smart Search")
about_button = st.sidebar.button("Chatbot QnA")
upload_button = st.sidebar.button("Upload Files")

# Determine which button was clicked and display the corresponding page
if home_button:
    st.session_state.page = 'smart search'
elif about_button:
    st.session_state.page = 'chatbot'
elif upload_button:
    st.session_state.page = 'Upload'


# Display the selected page
if st.session_state.page == 'smart search':
    home_page()
elif st.session_state.page == 'chatbot':
    chatbot(llm,
    st.session_state.casedb_retriever,
    st.session_state.callsdb_retriever,
    st.session_state.message
    )
elif st.session_state.page == 'Upload':
    upload()

llm.close()
