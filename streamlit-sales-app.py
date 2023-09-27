import streamlit as st

st.set_page_config(layout="wide")


#adding a single-line text input widget

# Define a function for each page
def home_page():
    import pandas as pd

    project_df = pd.read_excel('/home/talha/Downloads/PS - Competencies Management(1).xlsx',sheet_name='Proj-details')
    employee_df = pd.read_excel('/home/talha/Downloads/PS - Competencies Management(1).xlsx',sheet_name='Res-skills')
    domain_df = pd.read_excel('/home/talha/Downloads/PS - Competencies Management(1).xlsx',sheet_name='Proj. Dashboard',skiprows=1)

    unnamed_columns = [col for col in project_df.columns if col.startswith('Unnamed:')]
    named_columns = ['Missing Projects (267)','Level']
    project_df = project_df.drop(columns=unnamed_columns+named_columns)

    unnamed_columns = [col for col in employee_df.columns if col.startswith('Unnamed:')]
    named_columns = ['Type','Category']
    employee_df = employee_df.drop(columns=unnamed_columns+named_columns)



    st.title('Smart Search')
    skill = st.text_input('Enter Skill: ', 'e.g .NET OR .net')


    if skill == 'e.g .NET OR .net':
        skill= None

    client = st.text_input('Enter client name: ', 'Pepsi')

    if client == 'Pepsi':
        client= None
    #displaying the entered text

    if skill is None and client is None:
        st.write("")

    elif skill is not None and client is not None:
        st.write("Please ENTER Only one at a time")

    elif skill:
        st.write('Data matches with skill : ', skill)
        project_df.dropna(subset=['Competency (338)'],inplace=True)

        mask = project_df['Competency (338)'].str.contains(skill)

        # # Use the mask to filter the DataFrame
        result_df = project_df[mask]

        # Add a button to show all rows
        st.markdown("---")
        # st.sidebar.title(':blue[Project Details]')
        st.header('_Projects Matched_')
        if st.button('Show All Projects'):
            st.dataframe(result_df)

        else:
            st.dataframe(result_df.head(5))


        projects = result_df['Project'].unique().tolist()

        employee_df.dropna(subset=['Competency (398)'],inplace=True)

        mask = employee_df['Competency (398)'].str.contains(skill)

        # # Use the mask to filter the DataFrame
        result_df = employee_df[mask]

        # st.sidebar.title(':blue[Employee Details]')
        st.markdown("---")
        st.header('Employees Matched')
        if st.button('Show All Employees'):
            st.dataframe(result_df)
        else:
            st.dataframe(result_df.head(5))


        result_df = domain_df[domain_df['Project'].isin(projects)]
        # st.sidebar.title(':blue[Project Domain Details]')
        st.markdown("---")
        st.header('Domains Matched')

        if st.button('Show All Domains'):
            st.dataframe(result_df)
        else:
            st.dataframe(result_df.head(5))  

    elif client:
        # st.set_page_config(layout="wide")
        st.write('Data matches with client : ', client)

        domain_df.dropna(subset=['Project'],inplace=True)

        mask = domain_df['Project'].str.contains(client)

        # # Use the mask to filter the DataFrame
        result_df = domain_df[mask]
        # st.sidebar.title(':blue[Client Details]')
        st.markdown("---")
        st.header('Client Matched')

        if st.button('Client Detail'):
            st.dataframe(result_df)
        else:
            st.dataframe(result_df.head(5))  


def about_page():

    import sys
    import os
    import langchain
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import Docx2txtLoader
    from langchain.document_loaders import TextLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain,RetrievalQA
    from langchain.memory import ConversationBufferMemory
    from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from dotenv import load_dotenv

    from langchain.schema import (
        SystemMessage,
        HumanMessage,
        AIMessage
    )

    from streamlit_chat import message

    


    load_dotenv()
        
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # os.environ["OPENAI_API_KEY"] = "sk-"


    prompt_template = """you are helpful Ai assisstant that is helping sales people of software company for sales enablement you have employee profiles and project details and project case studies. 
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, 
    don't try to make up an answer.

    {context}

    Question: {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}


    documents=[]
    paths = ["profiles/","projects/","case_studies/"]
    for path in paths:
        for file in os.listdir(path):
            if file.endswith(".pdf"):
                pdf_path = path + file
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                doc_path = path + file
                loader = Docx2txtLoader(doc_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                text_path = path + file
                loader = TextLoader(text_path)
                documents.extend(loader.load())
        
    # print(documents)
    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
    vectordb.persist()
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
    # memory = ConversationBufferMemory(llm=llm, max_token_limit=100)
    retriever = vectordb.as_retriever( search_kwargs={'k': 3})

    qa=RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)



    st.header("Sales ChatBot🤖")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []


    # container for chat history
    response_container = st.container()
    # container for text box
    textcontainer = st.container()


    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("typing..."):
                response = qa({'query':query})['result']
            st.session_state.requests.append(query)
            st.session_state.responses.append(response) 
    with response_container:
        if st.session_state['responses']:

            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Create buttons for page selection in the sidebar
home_button = st.sidebar.button("Smart Search")
about_button = st.sidebar.button("Chatbot QnA")

# Determine which button was clicked and display the corresponding page
if home_button:
    st.session_state.page = 'Home'
elif about_button:
    st.session_state.page = 'About'


# Display the selected page
if st.session_state.page == 'Home':
    home_page()
elif st.session_state.page == 'About':
    about_page()
