import os
import constants
from tqdm import tqdm
from utils import  DataProcessor

from dotenv import load_dotenv

from create_db import DatabaseManager
db_manager = DatabaseManager()
db_manager.create_tables()
db_manager.close_connection()

def start_data_ingestion():

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

    if os.path.exists(constants.data) and os.path.isdir(constants.data):
        print('Variables Initializing...')

        data_processor = DataProcessor()

        print('Data ingestion process has been started... PLease wait. ')
        cwd = os.getcwd()
        paths = [os.path.join(cwd,constants.data)]

        for path in paths:
            for file in tqdm(os.listdir(path)):
                data_processor.processFile(file)

        data_processor.close()



    else:
        print('Data ingestion process has been Paused... PLease Check your data path. ')
    






# Example usage:
if __name__ == '__main__':
    start_data_ingestion()