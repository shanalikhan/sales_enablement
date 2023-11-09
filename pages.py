import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from utils import DataProcessor
import constants


load_dotenv()

if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    raise ValueError("OPENAI_API_KEY is not set")
else:
    print("OPENAI_API_KEY is set")

import openai

os.makedirs(constants.DATA, exist_ok=True)

# Define a function for each page
def home_page():
    """ Smart Search funtion"""
    project_df = pd.read_excel(constants.FILE_NAME,sheet_name='Proj-details')
    employee_df = pd.read_excel(constants.FILE_NAME,sheet_name='Res-skills')
    domain_df = pd.read_excel(constants.FILE_NAME,sheet_name='Proj. Dashboard',skiprows=1)

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

def upload_handler(file):
    """ uploader function used for data ingestion"""
    data_processor = DataProcessor()

    file_path = os.path.join(os.getcwd(),constants.DATA, file.name)

    if file:
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        st.success(f"File {file.name} processed and saved to directory {constants.DATA}!")

        data_processor.process_file(file.name, openai)
        data_processor.close()

def upload():
    """ function used to for upload section of app"""
    st.title("File Upload for Case Studies and sales Calls")

    # Upload widget for case studies
    st.subheader("Upload a Case Study (PDF,TEXT,DOCS) only")
    case_study_file = st.file_uploader("Choose a video file", type=["txt", "pdf", "docs"])
    if case_study_file:
        upload_handler(case_study_file)

    # Upload widget for calls
    st.subheader("Upload a sales Call (Video or MP3 audio)")
    call_file = st.file_uploader(
        "Choose a video or audio file",
        type=["mp4", "mkv", "avi", "mov", "mp3"]
        )
    if call_file:
        upload_handler(call_file)

def chatbot(llm, casedb_retriever, callsdb_retriever, message):
    """Function used for chatbot feature"""
    try:
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

                    response = llm.qa(query, casedb_retriever,callsdb_retriever,openai)

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
