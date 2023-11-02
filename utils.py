import pandas as pd
import re
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from create_db import DatabaseManager

class DataProcessor:
    def __init__(self, db_name='entities.db'):
        self.db_manager = DatabaseManager(db_name)

    @staticmethod
    def get_prompt_template():
        prompt_template = """you are helpful Ai assistant that is helping sales people of software company for sales enablement you have employee profiles and project details and project case studies. 
        Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, output should be in json where there will be two keys one is boolean named as found that will be true if you found the answer else will be false and second key named as answer will be your response if there is any.
        don't try to make up an answer.

        {context}

        Question: {question}
        """
        return prompt_template

    @staticmethod
    def load_dataframes(file_name = 'PS - Competencies Management.xlsx'):
        domain_df = pd.read_excel(file_name, sheet_name='Proj. Dashboard', skiprows=1)
        project_df = pd.read_excel(file_name, sheet_name='Proj-details')
        client_df = pd.read_excel(file_name, sheet_name='KM')
        invoices_df = pd.read_excel(file_name, sheet_name='invoices')

        domain_df.dropna(subset=['Project'], inplace=True)
        project_df.dropna(subset=['Project'], inplace=True)
        client_df.dropna(subset=['Project', 'Client'], inplace=True)
        invoices_df.dropna(subset=['project', 'client'], inplace=True)

        client_df['Project'] = client_df['Project'].apply(lambda x: x.split('-')[0])
        invoices_df['project'] = invoices_df['project'].apply(lambda x: x.split('-')[0])

        rule_df = pd.concat([domain_df['Project'], project_df['Project'], client_df['Project'], client_df['Client'], invoices_df['project'], invoices_df['client']], axis=0)

        rule_df = rule_df.unique()
        return rule_df

    @staticmethod
    def get_text_from_file(file, path):
        if file.endswith(".pdf"):
            pdf_path = path + file
            loader = PyPDFLoader(pdf_path)
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = path + file
            loader = Docx2txtLoader(doc_path)
        elif file.endswith('.txt'):
            text_path = path + file
            loader = TextLoader(text_path)
        return loader

    def get_entities_and_dump(self, data, comprehend, ner_threshold = 0.90):

        for metadatas, documents in tqdm(zip(data['metadatas'], data['documents'])):
            try:
                reference_document = metadatas['source']
                doc_name = "_".join(reference_document.split('_')[1:])
                if doc_name == '':
                    doc_name = reference_document
                self.db_manager.insert_into_documents(doc_name, metadatas['source'])
                # Extract entities here
                response = comprehend.detect_entities(Text=documents, LanguageCode='en')

                for dic in response['Entities']:
                    if dic['Type'] == 'ORGANIZATION' and dic['Score'] >= ner_threshold:
                        self.db_manager.insert_into_entities(dic['Text'], reference_document)
            except Exception as e:
                print(e)

    @staticmethod
    def get_sentence_split(text_list, file):
        jump = 20
        for index, i in enumerate(range(0, len(text_list), jump)):
            file_index = str(index) + '_' + file
            check = i + jump
            if check > len(text_list):
                splitted_text = '.'.join(text_list[i:])
            else:
                splitted_text = '.'.join(text_list[i:check])
        return splitted_text, file_index

    def apply_masking(self, text, rule_df, metadata):
        text = re.sub(r'\\n', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('•', '')

        replacement = 'xyz'
        rows = self.db_manager.get_entities_from_table(metadata['source'])
        ents = [row[1] for row in rows if row[1] != '']

        for row in rule_df:
            pattern = re.compile(r'(?<!\w)' + re.escape(row) + r'(?!\w)', re.IGNORECASE)
            text = pattern.sub(replacement, text)

        for entity in set(ents):
            entity = entity.replace('•', '')
            pattern = re.compile(r'(?<!\w)' + re.escape(entity) + r'(?!\w)', re.IGNORECASE)
            text = pattern.sub(replacement, text)

        return text

    def close(self):
        self.db_manager.close_connection()

# Example usage:
if __name__ == '__main__':
    data_processor = DataProcessor()
    # ... use data_processor methods as needed ...
    data_processor.close()