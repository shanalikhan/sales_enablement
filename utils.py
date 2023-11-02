import pandas as pd
import re
import os
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from create_db import DatabaseManager

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
### NER AWS COMPREHEND
import boto3
session = boto3.Session(profile_name='AE')


from moviepy.editor import *
from pydub.utils import mediainfo

import whisper
import openai
import json
from langchain.prompts import PromptTemplate
import constants

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
        splits_documents = []

        for index, i in enumerate(range(0, len(text_list), jump)):
            file_index = str(index) + '_' + file
            check = i + jump
            if check > len(text_list):
                splitted_text = '.'.join(text_list[i:])
            else:
                splitted_text = '.'.join(text_list[i:check])

            splits_documents.append(Document(page_content=splitted_text, metadata={"source": file_index}))

        return splits_documents

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

class DocumentProcessor:
    def __init__(self, db_name='entities.db'):
        self.db_manager = DatabaseManager(db_name)

    
    def close(self):
        self.db_manager.close_connection()

class ChromaManager:
    def __init__(self):
        self.comprehend = session.client('comprehend')

    def create_vector_db(self, db_directory, paths, data_processor, constants):
        if os.path.exists(db_directory) and os.path.isdir(db_directory):
            vectordb = Chroma(persist_directory=db_directory,embedding_function=OpenAIEmbeddings())
            
        else:
            documents=[]
            
            for path in paths:
                for file in tqdm(os.listdir(path)):
                    
                    loader = data_processor.get_text_from_file(file,path)
                    text_list = loader.load()[0].page_content.split('.')
                    documents.extend(data_processor.get_sentence_split(text_list, file))

            vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(),persist_directory=db_directory)
            vectordb.persist()
            vectordb_data  = vectordb.get()

            data_processor.get_entities_and_dump(vectordb_data, self.comprehend, constants.ner_threshold)

        retriever = vectordb.as_retriever( search_kwargs={'k': 3})

        return retriever 

    def update_vector_db(self, db_directory, paths, data_processor, constants):
        if os.path.exists(db_directory) and os.path.isdir(db_directory):
            vectordb = Chroma(persist_directory=db_directory,embedding_function=OpenAIEmbeddings())
            documents=[]
            
            for path in paths:
                for file in tqdm(os.listdir(path)):
                    
                    loader = data_processor.get_text_from_file(file,path)
                    text_list = loader.load()[0].page_content.split('.')
                    documents.extend(data_processor.get_sentence_split(text_list, file))
            
            vectordb.add_documents(documents)
            vectordb.persist()

        retriever = vectordb.as_retriever( search_kwargs={'k': 3})

        return retriever 

class AudioProcessor:
    def __init__(self):
        self.model = whisper.load_model("large")

    def convert_mp4_to_mp3(self, video_path, output_path):
        try:
            # Load video using moviepy
            video = VideoFileClip(video_path)
            
            # Extract audio from video
            audio = video.audio
            
            # Export audio as mp3
            audio.write_audiofile(output_path, codec='mp3')
            
            # Close the clips
            video.close()
            audio.close()
            return True
        except Exception as e:
            return False
    
    def transcribe_audio(self,audio_path):
        try:
            output_path = audio_path

            result = self.model.transcribe(audio_path,language="en")
            text  = result['text']

            f = open(output_path.replace('mp3','txt'),'w')
            f.write(str(text))
            f.close()
            print('file updated:',file)

            return True
        except Exception as e:
            print(e)
            return False

    def check_audio_channels(audio_path):
        # Get audio information using pydub's mediainfo
        info = mediainfo(audio_path)
        
        # Return the number of channels
        return int(info['channels']) 

class LLM:
    def __init__(self):
        self.data_processor = DataProcessor()

        self.prompt = PromptTemplate(
            template=self.data_processor.get_prompt_template(), input_variables=["context", "question"]
        )
        self.rule_df = self.data_processor.load_dataframes(constants.file_name)

    def QA(self, query, casedb_retriever, callsdb_retriever):
        
            docs = casedb_retriever.get_relevant_documents(
                query
            )
            context = ''
            for i, text in enumerate(docs):
                
                text = self.data_processor.apply_masking(text.page_content, self.rule_df, text.metadata)
                
                context += text
            case_prompt = self.prompt.format(context=context, question=query)
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": case_prompt}])
            response = json.loads(completion.choices[0].message.content)
            if response['found'] == False:
                docs = callsdb_retriever.get_relevant_documents(
                    query
                )
                context = ''
                for i, text in enumerate(docs):

                    text = self.data_processor.apply_masking(text.page_content, self.rule_df, text.metadata)

                    context += text
                calls_prompt = self.prompt.format(context=context, question=query)
                
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": calls_prompt}])
                response = json.loads(completion.choices[0].message.content)
                try:
                    response=response['answer']
                except:
                    response = 'i am not able to answer right now please try again'
            else:
                response = response['answer']
            return response
    def close(self):
        self.data_processor.close()
    

# Example usage:
if __name__ == '__main__':
    data_processor = DataProcessor()
    # ... use data_processor methods as needed ...
    data_processor.close()