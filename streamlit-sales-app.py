import streamlit as st

# Check if the data is already loaded
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Load the data only if it's not already loaded
if not st.session_state.data_loaded:
    import sys
    import os
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import Docx2txtLoader
    from langchain.document_loaders import TextLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from dotenv import load_dotenv

    from langchain.schema import (
        SystemMessage,
        HumanMessage,
        AIMessage
    )

    from streamlit_chat import message

    import pandas as pd
    import spacy
    nlp = spacy.load('en_core_web_sm')
    
    
    domain_df = pd.read_excel('PS - Competencies Management.xlsx',sheet_name='Proj. Dashboard',skiprows=1)
    project_df = pd.read_excel('PS - Competencies Management.xlsx',sheet_name='Proj-details')
    client_df = pd.read_excel('PS - Competencies Management.xlsx',sheet_name='KM')
    invoices_df = pd.read_excel('PS - Competencies Management.xlsx',sheet_name='invoices')

    domain_df.dropna(subset=['Project'],inplace=True)
    project_df.dropna(subset=['Project'],inplace=True)
    client_df.dropna(subset=['Project','Client'],inplace=True)
    invoices_df.dropna(subset=['project','client'],inplace=True)

    client_df['Project']=client_df['Project'].apply(lambda x: x.split('-')[0])
    invoices_df['project']=invoices_df['project'].apply(lambda x: x.split('-')[0])


    rule_df = pd.concat([domain_df['Project'], project_df['Project'], client_df['Project'],client_df['Client'], invoices_df['project'],invoices_df['client']], axis=0)


    rule_df=rule_df.unique()

    load_dotenv()

    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # os.environ["OPENAI_API_KEY"] = "sk-"


    prompt_template = """you are helpful Ai assisstant that is helping sales people of software company for sales enablement you have employee profiles and project details and project case studies. 
    Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, output should be in json where threre will be two keys one is boolean named as found that will be true if you found the answer else will be false and second key named as answer will be your response if there is any.
    don't try to make up an answer.

    {context}

    Question: {question}
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    text_splitter = text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 50
    )

    db_directory = 'case_study_db'
    db_directory2 = 'calls_db'

    if os.path.exists(db_directory) and os.path.isdir(db_directory):
        casedb = Chroma(persist_directory=db_directory,embedding_function=OpenAIEmbeddings())
        
    else:
        documents=[]
        paths = ["projects/","case_studies/"]
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
                    
        casedb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(),persist_directory=db_directory)
        casedb.persist()
    
    casedb_retriever = casedb.as_retriever( search_kwargs={'k': 3})
    
    if os.path.exists(db_directory2) and os.path.isdir(db_directory2):
        callsdb = Chroma(persist_directory=db_directory2,embedding_function=OpenAIEmbeddings())
        
    else:
        documents=[]
        paths = ["tkxel_cals/"]
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
        documents = text_splitter.split_documents(documents)
        callsdb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(),persist_directory=db_directory2)
        callsdb.persist()
    
    callsdb_retriever = callsdb.as_retriever( search_kwargs={'k': 3})
    
    
    
    st.session_state.casedb_retriever = casedb_retriever
    st.session_state.callsdb_retriever = callsdb_retriever
    st.session_state.prompt = prompt
    st.session_state.message = message
    st.session_state.rule_df = rule_df
    st.session_state.nlp = nlp
    st.session_state.data_loaded = True



st.set_page_config(layout="wide")


#Maksing funtion
def apply_masking(text,rule_df,nlp):
    import re
    
    text = re.sub(r'\\n', '', text)
    text = re.sub(r'\s+', ' ', text)
    text=text.replace('•','')
    
    doc = nlp(text)
    replacement = 'xyz'

    ents = []
    for e in doc.ents:
        if e.label_ == 'ORG':
            ents.append(e.text)

    for row in rule_df:
        # Create a regular expression pattern to match standalone substrings with case insensitivity
        pattern = re.compile(r'(?<!\w)' + re.escape(row) + r'(?!\w)', re.IGNORECASE)

        # Use the sub() method to replace all occurrences of the standalone substring
        text = pattern.sub(replacement, text)

        
    # text=text.replace('•','')
    for entity in set(ents):
        entity = entity.replace('•','')

        # Create a regular expression pattern to match standalone substrings with case insensitivity
        pattern = re.compile(r'(?<!\w)' + re.escape(entity) + r'(?!\w)', re.IGNORECASE)

        # Use the sub() method to replace all occurrences of the standalone substring
        text = pattern.sub(replacement, text)

    # print(set(ents))
    return text

# Define a function for each page
def home_page():
    import pandas as pd

    project_df = pd.read_excel('PS - Competencies Management.xlsx',sheet_name='Proj-details')
    employee_df = pd.read_excel('PS - Competencies Management.xlsx',sheet_name='Res-skills')
    domain_df = pd.read_excel('PS - Competencies Management.xlsx',sheet_name='Proj. Dashboard',skiprows=1)

    unnamed_columns = [col for col in project_df.columns if col.startswith('Unnamed:')]
    named_columns = ['Missing Projects (267)','Level']
    project_df = project_df.drop(columns=unnamed_columns+named_columns)

    unnamed_columns = [col for col in employee_df.columns if col.startswith('Unnamed:')]
    named_columns = ['Type','Category']
    employee_df = employee_df.drop(columns=unnamed_columns+named_columns)



    st.title('Smart Search')
    skill = st.text_input('Enter Skill: ', 'e.g .NET OR .net')


    if skill == 'e.g .NET OR .net' or skill == '':
        skill= None

    client = st.text_input('Enter client name: ', 'Pepsi')

    if client == 'Pepsi' or client == '':
        client= None
    #displaying the entered text
    

    if skill is None and client is None:
        st.write("")

    elif skill is not None and client is not None:
        st.write("Please ENTER Only one at a time")
        print(skill)
        print(client)

    elif skill:
        st.write('Data matches with skill : ', skill)
        project_df.dropna(subset=['Competency (338)'],inplace=True)

        mask = project_df['Competency (338)'].str.contains(skill,case=False)

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

        mask = employee_df['Competency (398)'].str.contains(skill,case=False)

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

        mask = domain_df['Project'].str.contains(client,case=False)

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
    import openai
    import json
    
    
    casedb_retriever = st.session_state.casedb_retriever
    callsdb_retriever = st.session_state.callsdb_retriever
    prompt = st.session_state.prompt
    message = st.session_state.message
    nlp = st.session_state.nlp
    
    rule_df = st.session_state.rule_df
    
    replacement = 'xyz'
    
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
                docs = casedb_retriever.get_relevant_documents(
                    query
                )
                context = ''
                for i, text in enumerate(docs):
                    
                    text = apply_masking(text.page_content,rule_df,nlp)
                    
                    context += text
                case_prompt = prompt.format(context=context, question=query)
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": case_prompt}])
                response = json.loads(completion.choices[0].message.content)
                if response['found'] == False:
                    docs = callsdb_retriever.get_relevant_documents(
                        query
                    )
                    context = ''
                    for i, text in enumerate(docs):

                        text = apply_masking(text.page_content,rule_df,nlp)

                        context += text
                    prompt = prompt.format(context=context, question=query)
                    
                    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
                    response = json.loads(completion.choices[0].message.content)
                    try:
                        response=response['answer']
                    except:
                        response = 'i am not able to answer right now please try again'
                else:
                    response = response['answer']
                    
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
