""" file to ingest data into vector dbs and entity db"""
import os
from dotenv import load_dotenv
from tqdm import tqdm
import constants
from utils import  DataProcessor
from create_db import DatabaseManager

def start_data_ingestion():
    """" function that takes directory and put in db"""

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

    if os.path.exists(constants.DATA) and os.path.isdir(constants.DATA):
        print('Variables Initializing...')

        data_processor = DataProcessor()

        print('Data ingestion process has been started... PLease wait. ')
        cwd = os.getcwd()
        paths = [os.path.join(cwd,constants.DATA)]

        for path in paths:
            for file in tqdm(os.listdir(path)):
                data_processor.process_file(file)

        data_processor.close()



    else:
        print('Data ingestion process has been Paused... PLease Check your data path. ')
    

# Example usage:
if __name__ == '__main__':
    start_data_ingestion()