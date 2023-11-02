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
    
    
    
    from langchain.prompts import PromptTemplate
    from langchain.schema import (
        SystemMessage,
        HumanMessage,
        AIMessage
    )

    from dotenv import load_dotenv

    from streamlit_chat import message

    import constants

    ### Preprodata_processorcessors ###
    from utils import DataProcessor, ChromaManager
    data_processor = DataProcessor()
    chroma_manager = ChromaManager()
    ### db manager ###
    from create_db import DatabaseManager
    db_manager = DatabaseManager()
    db_manager.create_tables()
    db_manager.close_connection()


    load_dotenv()
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # os.environ["OPENAI_API_KEY"] = "sk-"

    
    print('Case studies data processing started -------------------------------------------------->')
    
    paths = [constants.projects_path, constants.case_studies_path]
    casedb_retriever = chroma_manager.create_vector_db(constants.db_directory, paths, data_processor, constants)

    print('Audio data processing started -------------------------------------------------->')

    paths = [constants.calls_path]
    callsdb_retriever = chroma_manager.create_vector_db(constants.db_directory2, paths, data_processor, constants)


    
    
    
    st.session_state.casedb_retriever = casedb_retriever
    st.session_state.callsdb_retriever = callsdb_retriever
    st.session_state.message = message
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

def upload_handler(file, calls=False):
    import os
    from utils import DataProcessor, ChromaManager, AudioProcessor
    import constants
    import shutil
    from tqdm import tqdm

    direc_name = constants.upload_directory
    data_processor = DataProcessor()
    chroma_manager = ChromaManager()
    audio_processor = AudioProcessor()

    file_path = os.path.join(os.getcwd(),direc_name, file.name)
    if file.name.endswith('txt') or file.name.endswith('pdf') or file.name.endswith('docs'):
        try:
            source = os.path.join(constants.upload_directory,file.name)
            destination = constants.case_studies_path
            with open(file_path, "wb") as f:
                f.write(file.getvalue())

            paths = [constants.upload_casestudies_path]
            
            casedb_retriever = chroma_manager.update_vector_db(constants.db_directory, paths, data_processor, constants)
            
            st.session_state.casedb_retriever = casedb_retriever
            
            shutil.move(source, destination)
            
            st.success(f"File {file.name} saved!")
        except Exception as e:
            print(e)
            st.success(f"File {file.name}not able to save or processed!")
    else:
        try:
            source = os.path.join(constants.upload_directory,file.name)
            destination = constants.calls_path
            
            with open(file_path, "wb") as f:
                f.write(file.getvalue())

            paths = [constants.upload_casestudies_path]
            print('here')

            for path in paths:
                for file in tqdm(os.listdir(path)):
                    if file.endswith('.mp4'):
                        mp3_path = file_path.replace('.mp4','.mp3')
                        if audio_processor.convert_mp4_to_mp3(file_path,mp3_path):
                            print('\n\naudio processsed \n\n')
                            if audio_processor.transcribe_audio(mp3_path):
                                print('Done')
                                try:
                                    os.remove(mp3_path)
                                    print(f"The file {mp3_path} has been removed successfully")
                                    os.remove(file_path)
                                    print(f"The file {mp3_path} has been removed successfully")
                                except FileNotFoundError:
                                    print("The file does not exist")
                                except PermissionError:
                                    print("You do not have permission to delete this file")
                                except Exception as e:
                                    print(f"An error occurred: {e}")
            callsdb_retriever = chroma_manager.update_vector_db(constants.db_directory2, paths, data_processor, constants)
            
            st.session_state.callsdb_retriever = callsdb_retriever
            
            shutil.move(source, destination)
            
            st.success(f"File {file.name} saved!")
        except Exception as e:
            print(e)
            st.success(f"File {file.name}not able to save or processed!")




def upload():
    st.title("File Upload for Case Studies and sales Calls")

    # Upload widget for case studies
    st.subheader("Upload a Case Study (PDF,TEXT,DOCS) only")
    case_study_file = st.file_uploader("Choose a video file", type=["txt", "pdf", "docs"])
    if case_study_file:
        upload_handler(case_study_file)

    # Upload widget for calls
    st.subheader("Upload a sales Call (Video or MP3 audio)")
    call_file = st.file_uploader("Choose a video or audio file", type=["mp4", "mkv", "avi", "mov", "mp3"])
    if call_file:
        upload_handler(call_file)

def chatbot():
    import openai
    import json
    from utils import LLM
    llm = LLM()    
    try:
        
        
        casedb_retriever = st.session_state.casedb_retriever
        callsdb_retriever = st.session_state.callsdb_retriever
        message = st.session_state.message
        
        
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
                    response = llm.QA(query, casedb_retriever,callsdb_retriever)
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
        llm.close()

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
    
