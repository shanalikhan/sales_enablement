'''utils file for AE app'''
from moviepy.editor import VideoFileClip
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from speechbrain.pretrained import VAD
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from create_db import DatabaseManager
from pydub import AudioSegment
import torchaudio
import pandas as pd
from tqdm import tqdm
import uuid
import boto3
import whisper
import json
import constants
import yaml
import re
import os
import shutil

session = boto3.Session(profile_name='AE')


class DataProcessor:
    """ Class used to process the data"""

    def __init__(self, db_name='entities.db'):
        self.db_manager = DatabaseManager(db_name)
        self.comprehend = session.client('comprehend')
        self.audio_processor = AudioProcessor()

    @staticmethod
    def get_prompt_template():
        """ function that set the prompt template used by GPT """

        prompt_template = """you are helpful Ai assistant that is helping
        sales people of software company for sales enablementyou have employee 
        profiles and project details and project case studies. Use the following 
        context to answer the question at the end. If you don't know the answer, 
        just say that you don't know, output should be in json where there will 
        be two keys one is boolean named as found that will be true if you found 
        the answer else will be false and second key named as answer will be your 
        response if there is any. don't try to make up an answer.

        {context}

        Question: {question}
        """
        return prompt_template

    @staticmethod
    def load_dataframes(file_name = 'PS - Competencies Management.xlsx'):
        """ Function that used to read all data from sheets for masking """

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

        rule_df = pd.concat([domain_df['Project'], project_df['Project'],
        client_df['Project'], client_df['Client'],
        invoices_df['project'], invoices_df['client']],
        axis=0)

        rule_df = rule_df.unique()
        return rule_df

    @staticmethod
    def get_text_from_file(file_path):
        """ function takes file and convert those into text document"""

        if file_path.endswith(".pdf"):
            pdf_path = file_path
            loader = PyPDFLoader(pdf_path)
        elif file_path.endswith('.docx') or file_path.endswith('.doc'):
            doc_path = file_path
            loader = Docx2txtLoader(doc_path)
        elif file_path.endswith('.txt'):
            text_path = file_path
            loader = TextLoader(text_path)
        return loader

    def get_entities_and_dump(self, data, comprehend, ner_threshold = 0.90):
        """ function takes documents and put them into db with their respective entities"""

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
        """ Function used to split sentences for embeddings"""

        jump = 20
        splits_documents = []

        for index, i in enumerate(range(0, len(text_list), jump)):
            file_index = str(index) + '_' + file
            check = i + jump
            if check > len(text_list):
                splitted_text = '.'.join(text_list[i:])
            else:
                splitted_text = '.'.join(text_list[i:check])
            obj = Document(page_content=splitted_text, metadata={"source": file_index})
            splits_documents.append(obj)

        return splits_documents

    def apply_masking(self, text, rule_df, metadata):
        """ Funciton that mask entites from sqlite db and sheets"""

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

    def process_file(self,file, openai):
        """ Function is the part of data ingestion and takes file by file"""
        chroma_manager = ChromaManager()
        os.makedirs(constants.DATA, exist_ok=True)
        file_path = os.path.join(os.getcwd(),constants.DATA, file)

        if file.endswith('txt') or file.endswith('pdf') or file.endswith('docs'):
            loader = self.get_text_from_file(file_path)
            text_list = loader.load()[0].page_content.split('.')
            list_of_documents = self.get_sentence_split(text_list,file)

            chroma_manager.put_in_vectordb(constants.CASE_DB, list_of_documents)

            vectordb_dict = {'metadatas':[],'documents':[]}

            for doc in list_of_documents:
                vectordb_dict['documents'].append(doc.page_content)
                vectordb_dict['metadatas'].append(doc.metadata)

            self.get_entities_and_dump(vectordb_dict, self.comprehend, constants.NER_THRESHOLD)

            # print('file data dump in vector db and entities db')

        elif file.endswith('mp4'):
            status, mp3_path = self.audio_processor.convert_mp4_to_mp3(
                file_path,
                file_path.replace('mp4','mp3')
                )

            if status:
                print('MP3 convervsion successfull')
                os.remove(file_path)
                print('mp4 file remvoed successfully')
                
                text = self.audio_processor.transcribe_audio(mp3_path, openai)
                text_list = text.split('.')
                list_of_documents = self.get_sentence_split(text_list,file)

                chroma_manager.put_in_vectordb(constants.CALLS_DB, list_of_documents)

                vectordb_dict = {'metadatas':[],'documents':[]}

                for doc in list_of_documents:
                    vectordb_dict['documents'].append(doc.page_content)
                    vectordb_dict['metadatas'].append(doc.metadata)

                self.get_entities_and_dump(vectordb_dict, self.comprehend, constants.NER_THRESHOLD)

                print('file data dump in vector db and entities db')
            else:
                print('Three is issue with the video file please check')

        elif file.endswith('mp3'):
            text = self.audio_processor.transcribe_audio(file_path, openai)
            # print(text)
            text_list = text.split('.')
            list_of_documents = self.get_sentence_split(text_list,file)

            chroma_manager.put_in_vectordb(constants.CALLS_DB, list_of_documents)

            vectordb_dict = {'metadatas':[],'documents':[]}

            for doc in list_of_documents:
                vectordb_dict['documents'].append(doc.page_content)
                vectordb_dict['metadatas'].append(doc.metadata)

            self.get_entities_and_dump(vectordb_dict, self.comprehend, constants.NER_THRESHOLD)

            print('file data dump in vector db and entities db')
        else:
            print('Three is issue with the audio file please check')

    def close(self):
        """ function used to disable the db connection"""

        self.db_manager.close_connection()

class ChromaManager:
    """ Class use to dump embedding into db and initialize the vectordb objects"""

    def __init__(self):
        pass

    def put_in_vectordb(self, db_directory, documents):
        """ Function used put data into vectordb"""
        try:
            if os.path.exists(db_directory) and os.path.isdir(db_directory):
                vectordb = Chroma(
                    persist_directory=db_directory,
                    embedding_function=OpenAIEmbeddings())

                vectordb.add_documents(documents)

            else:
                vectordb = Chroma.from_documents(documents,
                    embedding=OpenAIEmbeddings(),
                    persist_directory=db_directory)

            vectordb.persist()
        except Exception as e:
            print(e)

    def get_retriever(self):
        """ Function used initialize the vectordb retireivers objects"""

        if os.path.exists(constants.CASE_DB) and os.path.isdir(constants.CASE_DB):
            vectordb = Chroma(
                persist_directory=constants.CASE_DB,
                embedding_function=OpenAIEmbeddings()
                )

            case_retriever = vectordb.as_retriever( search_kwargs={'k': 3})
            print('case study db retriver initialized')
        else:
            case_retriever = None

        if os.path.exists(constants.CALLS_DB) and os.path.isdir(constants.CALLS_DB):
            vectordb = Chroma(
                persist_directory=constants.CALLS_DB,
                embedding_function=OpenAIEmbeddings()
                )

            call_retriever = vectordb.as_retriever( search_kwargs={'k': 3})
            print('calls db retriver initialized')
        else:
            call_retriever = None

        return case_retriever, call_retriever


class AudioProcessor:
    """ Class used to perform audio video processing"""

    def __init__(self):
        with open(constants.CONFIG, 'r') as file:
            data = yaml.safe_load(file)
        self.mode = data['WhisperService']['mode']

        if self.mode == 'offline':
            self.model = whisper.load_model("small")
        else:
            self.VAD = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                savedir="pretrained_models/vad-crdnn-libriparty"
                )
            self.model= None

    def convert_mp4_to_mp3(self, video_path, output_path):
        """ Function used to convert mp4 video to mp3 audio"""
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
            return True, output_path
        except Exception as e:
            print(e)
            return False, None

    def transcribe_audio(self,audio_path,openai):
        """ Function used to transcribe the audio into text using whisper model"""
        try:
            if self.mode == 'offline':
                result = self.model.transcribe(audio_path,language="en")
                text  = result['text']
            else:
                df = pd.DataFrame(columns=['id', 'start','end', 'text'])
                channels = self.separate_channels(audio_path,
                constants.LEFT_AUDIO_PATH,
                constants.RIGHT_AUDIO_PATH,
                constants.CHUNKS_DIRECTORY
                )
                left_boundries, right_boundries = self.get_boundries(constants.LEFT_AUDIO_PATH,
                constants.RIGHT_AUDIO_PATH,
                channels,
                constants.CHUNKS_DIRECTORY)

                left_segments, right_segments = self.get_segments(constants.LEFT_AUDIO_PATH,
                constants.RIGHT_AUDIO_PATH,
                left_boundries,
                right_boundries,
                channels,
                constants.CHUNKS_DIRECTORY)

                if channels == 2:
                    df = self.save_segments(left_segments,
                    left_boundries,
                    df,
                    constants.CHUNKS_DIRECTORY
                    )
                    df = self.save_segments(right_segments,
                    right_boundries,
                    df,
                    constants.CHUNKS_DIRECTORY)
                else:
                    df = self.save_segments(left_segments,
                    left_boundries,
                    df,
                    constants.CHUNKS_DIRECTORY
                    )

                df['start'] = df['start'].astype(float)
                df['end'] = df['end'].astype(float)

                df['start'] = df['start'].round(2)
                df['end'] = df['end'].round(2)

                df.drop_duplicates(
                    subset=['start', 'end'],
                    keep='first',inplace=True
                    )
                df.sort_values(by='start',inplace=True)

                df['text'] = df['id'].apply(lambda x: self.audio_transcription_to_text(x, openai, constants.CHUNKS_DIRECTORY))
                text = df['text'].str.cat(sep=' ')
                try:
                    shutil.rmtree(constants.CHUNKS_DIRECTORY)
                    print(f"Directory '{constants.CHUNKS_DIRECTORY}' and all its contents have been removed")
                except OSError as error:
                    print(f"Error: {error}")
                    print(f"Failed to remove directory '{constants.CHUNKS_DIRECTORY}' and its contents")

            return text
        except Exception as e:
            print(e)
            return False

    def audio_transcription_to_text(self, file_id, openai, output_dir='audio_chunks'):
        """ Function used to transcribe the audio into text using whisper online model"""
        file_path = os.path.join(output_dir, str(file_id))
        file_path +='.wav'
        file = open(file_path, "rb")
        transcription = openai.Audio.transcribe("whisper-1", file, language='en')
        return transcription['text']

    def save_segments(self, segments, boundaries, df, output_dir='audio_chunks', sample_rate=16000):
        """
        Saves the segments as audio files.

        Arguments:
        segments -- List of tensors containing the audio segments.
        sample_rate -- The sample rate of the original audio file.
        output_dir -- Directory where the audio segments will be saved.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over the segments and save each one as a WAV file
        for i, (segment,duration) in tqdm(enumerate(zip(segments,boundaries))):
            random_uuid = uuid.uuid4()
            # Define the path for the output file
            output_file_path = os.path.join(output_dir, f"{random_uuid}.wav")
            
            df2 = pd.DataFrame([[random_uuid,f'{duration[0]}',f'{duration[1]}', '']], columns=['id', 'start','end', 'text'])
            
            df = pd.concat([df, df2], ignore_index=True)
            # Save the tensor as a WAV file
            torchaudio.save(output_file_path, segment, sample_rate)
        return df

    def separate_channels(self, audio_path, left_output_path, right_output_path, output_dir='audio_chunks', frame_rate= 16000):
        """ Function used to separate audio channels"""
        # Load the audio file using pydub
        os.makedirs(output_dir, exist_ok=True)

        audio = AudioSegment.from_mp3(audio_path)
        audio = audio.set_frame_rate(frame_rate)

        left_output_path = os.path.join(output_dir, left_output_path)
        right_output_path = os.path.join(output_dir, right_output_path)
        # Check if the audio is stereo
        if audio.channels == 2:
            # Split the stereo audio into its left and right channels
            channels = audio.split_to_mono()

            # Save the left and right channels to separate files
            channels[0].export(left_output_path, format="wav")
            channels[1].export(right_output_path, format="wav")

            print(f"Left channel saved to {left_output_path}")
            print(f"Right channel saved to {right_output_path}")

            return audio.channels
        else:
            audio.export(left_output_path, format="wav")
            print(f"Mono audio saved to {left_output_path}")
            return audio.channels
        return 0

    def get_boundries(self, left_output_path, right_output_path, channels, output_dir='audio_chunks'):
        """ Function used to get VAD boundries"""
        left_output_path = os.path.join(output_dir, left_output_path)
        right_output_path = os.path.join(output_dir, right_output_path)

        if channels == 2:
            left_boundaries = self.VAD.get_speech_segments(left_output_path,
            large_chunk_size=15,
            small_chunk_size=10
            )
            right_boundaries = self.VAD.get_speech_segments(right_output_path,
            large_chunk_size=15,
            small_chunk_size=10
            )
            return left_boundaries, right_boundaries
        else:
            left_boundaries = self.VAD.get_speech_segments(
                left_output_path,
                large_chunk_size=15,
                small_chunk_size=10
                )
            return left_boundaries, None

    def get_segments(self, left_output_path, right_output_path, left_boundaries,right_boundaries, channels, output_dir='audio_chunks'):
        """ Function used to get VAD boundries segments"""
        left_output_path = os.path.join(output_dir, left_output_path)
        right_output_path = os.path.join(output_dir, right_output_path)
        if channels == 2:
            left_segments = self.VAD.get_segments(left_boundaries,left_output_path)
            right_segments = self.VAD.get_segments(left_boundaries,right_output_path)
            return left_segments, right_segments
        else:
            left_segments = self.VAD.get_segments(left_boundaries,left_output_path)
            return left_segments, None
            
    def get_audio_info(self, audio_file_path):
        """ Function used to get audio detials"""
        audio = AudioSegment.from_file(audio_file_path)

        # Extract bit rate (in bits per second)
        bit_rate = audio.frame_rate * audio.frame_width * 8

        # Extract sample rate (in samples per second)
        sample_rate = audio.frame_rate

        # Extract number of channels
        channels = audio.channels

        return {
            "bit_rate": bit_rate,
            "sample_rate": sample_rate,
            "channels": channels
        }

class LLM:
    """ Class used for chat with LLM"""
    def __init__(self):
        self.data_processor = DataProcessor()

        self.prompt = PromptTemplate(
            template=self.data_processor.get_prompt_template(),
            input_variables=["context", "question"]
        )

        self.rule_df = self.data_processor.load_dataframes(constants.FILE_NAME)

    def qa(self, query, casedb_retriever, callsdb_retriever,openai):
        """ Function used to provide chat functionality"""
        try:
            docs = casedb_retriever.get_relevant_documents(
                query
            )
            context = ''
            for i, text in enumerate(docs):

                text = self.data_processor.apply_masking(
                    text.page_content,
                    self.rule_df,
                    text.metadata
                    )

                context += text

            case_prompt = self.prompt.format(context=context, question=query)

            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user",
                    "content": case_prompt
                    }
                    ])

            response = json.loads(completion.choices[0].message.content)
            if response['found'] == False:
                docs = callsdb_retriever.get_relevant_documents(
                    query
                )
                context = ''
                for i, text in enumerate(docs):

                    text = self.data_processor.apply_masking(
                        text.page_content,
                        self.rule_df, text.metadata
                        )

                    context += text
                calls_prompt = self.prompt.format(
                    context=context,
                    question=query
                    )

                completion = openai.ChatCompletion.create(
                    model = "gpt-4",
                    messages=[
                        {"role": "user",
                        "content": calls_prompt
                        }
                        ])

                response = json.loads(completion.choices[0].message.content)
                try:
                    response=response['answer']
                except Exception as e:
                    print(e)
                    response = 'i am not able to answer right now please try again'
            else:
                response = response['answer']
        except Exception as e:
            print(e)
            response = 'i am not able to answer right now please try again'
        return response
    def close(self):
        "Function used to close the db connection"
        self.data_processor.close()


# Example usage:
if __name__ == '__main__':
    data_processor = DataProcessor()
    # ... use data_processor methods as needed ...
    data_processor.close()
