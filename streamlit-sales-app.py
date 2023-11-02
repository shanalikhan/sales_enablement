import streamlit as st

# Check if the data is already loaded



if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


# Load the data only if it's not already loaded
if st.session_state.data_loaded == False:
    st.session_state.data_loaded = True
    import sys
    import os
    from tqdm import tqdm
    
    from langchain.docstore.document import Document
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.prompts import PromptTemplate
    from langchain.schema import (
        SystemMessage,
        HumanMessage,
        AIMessage
    )

    from dotenv import load_dotenv

    from streamlit_chat import message

    import constants
    ### NER AWS COMPREHEND
    import boto3
    session = boto3.Session(profile_name='AE')
    comprehend = session.client('comprehend')
    ### Preprodata_processorcessors ###
    from utils import DataProcessor
    data_processor = DataProcessor()
    ### db manager ###
    from create_db import DatabaseManager
    db_manager = DatabaseManager()
    db_manager.create_tables()
    db_manager.close_connection()

    rule_df = data_processor.load_dataframes(constants.file_name)

    load_dotenv()
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # os.environ["OPENAI_API_KEY"] = "sk-"


    prompt_template = data_processor.get_prompt_template()

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    
    print('Case studies data processing started -------------------------------------------------->')
    if os.path.exists(constants.db_directory) and os.path.isdir(constants.db_directory):
        casedb = Chroma(persist_directory=constants.db_directory,embedding_function=OpenAIEmbeddings())
        
    else:
        documents=[]
        paths = [constants.projects_path, constants.case_studies_path]
        for path in paths:
            for file in tqdm(os.listdir(path)):
                
                loader = data_processor.get_text_from_file(file,path)
                documents.extend(loader.load())

        casedb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(),persist_directory=constants.db_directory)
        casedb.persist()
        casedb_data  = casedb.get()

        data_processor.get_entities_and_dump(casedb_data, comprehend, constants.ner_threshold)

    casedb_retriever = casedb.as_retriever( search_kwargs={'k': 3})

    print('Audio data processing started -------------------------------------------------->')
    
    if os.path.exists(constants.db_directory2) and os.path.isdir(constants.db_directory2):
        callsdb = Chroma(persist_directory=constants.db_directory2,embedding_function=OpenAIEmbeddings())
        
    else:
        documents=[]
        paths = [constants.calls_path]
        jump = 20

        for path in paths:
            for file in tqdm(os.listdir(path)):

                loader = data_processor.get_text_from_file(file,path)
                    
                text_list = loader.load()[0].page_content.split('.')
                
                for index, i in enumerate(range(0, len(text_list), jump)):
                    file_index = str(index) + '_' + file
                    check = i + jump
                    if check > len(text_list):
                        splitted_text = '.'.join(text_list[i:])
                    else:
                        splitted_text = '.'.join(text_list[i:check])
                    documents.append(Document(page_content=splitted_text, metadata={"source": file_index}))

        callsdb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(),persist_directory=constants.db_directory2)
        callsdb.persist()
        callsdb_data  = callsdb.get()

        data_processor.get_entities_and_dump(callsdb_data, comprehend, constants.ner_threshold)
    
    callsdb_retriever = callsdb.as_retriever( search_kwargs={'k': 3})
    
    
    
    st.session_state.casedb_retriever = casedb_retriever
    st.session_state.callsdb_retriever = callsdb_retriever
    st.session_state.prompt = prompt
    st.session_state.message = message
    st.session_state.rule_df = rule_df
    st.session_state.data_loaded = True
    data_processor.close()

    # st.session_state.comprehend = comprehend



st.set_page_config(layout="wide")


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

def handle_case_study(file):
    import os
    # import PyPDF2
    # from docx import Document

    with open(os.path.join(os.getcwd(), file.name), "wb") as f:
        f.write(file.getvalue())
    st.success(f"File {file.name} saved!")
    

# Function to handle uploaded call videos or audios
def handle_call(file):
    import os
    # Check the file type to determine if it's a video or audio
    with open(os.path.join(os.getcwd(), file.name), "wb") as f:
        f.write(file.getvalue())
    st.success(f"File {file.name} saved!")

    if file.type == "audio/mp3":
        st.audio(file)
    else:
        st.video(file)
    


def upload():
    st.title("File Upload for Case Studies and sales Calls")

    # Upload widget for case studies
    st.subheader("Upload a Case Study (PDF,TEXT,DOCS) only")
    case_study_file = st.file_uploader("Choose a video file", type=["txt", "pdf", "docs"])
    if case_study_file:
        handle_case_study(case_study_file)

    # Upload widget for calls
    st.subheader("Upload a sales Call (Video or MP3 audio)")
    call_file = st.file_uploader("Choose a video or audio file", type=["mp4", "mkv", "avi", "mov", "mp3"])
    if call_file:
        handle_call(call_file)

def chatbot():
    try:
        import openai
        import json
        from utils import DataProcessor

        data_processor = DataProcessor()
        
        
        casedb_retriever = st.session_state.casedb_retriever
        callsdb_retriever = st.session_state.callsdb_retriever
        prompt = st.session_state.prompt
        message = st.session_state.message
        
        
        rule_df = st.session_state.rule_df
        
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
                        
                        text = data_processor.apply_masking(text.page_content, rule_df, text.metadata)
                        
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

                            text = data_processor.apply_masking(text.page_content, rule_df, text.metadata)

                            context += text
                        calls_prompt = prompt.format(context=context, question=query)
                        
                        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": calls_prompt}])
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
    except Exception as e:
        print(e)

    finally:
        data_processor.close()

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
    chatbot()
elif st.session_state.page == 'Upload':
    upload()
    
