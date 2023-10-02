import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pytesseract
from pdf2image import convert_from_path
from docx import Document
import PyPDF2
import openai
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from pymilvus import connections, db
from dotenv import load_dotenv
from second_layer_query import second_layer
import time,threading

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
if api_key is None:
    raise ValueError("OpenAI API key not found.")

openai.api_key = api_key

class ResumeParser:
    def __init__(self):
        print("start connecting to Milvus")
        connections.connect("default", host="127.0.0.1", port="19530")

        u_id = FieldSchema(name="Unique_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        file_name = FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=100)
        file_content = FieldSchema(name="file_content", dtype=DataType.VARCHAR, max_length=50000)
        cv_vector = FieldSchema(name="cv_vector", dtype=DataType.FLOAT_VECTOR, dim=1536)

        schema = CollectionSchema(
            fields=[u_id, file_name, file_content, cv_vector],
            description="Resume Parser"
        )
        collection_name = "Resume"

        self.collection = Collection(
            name=collection_name,
            schema=schema,
            using='default',
            shards_num=2,
            consistency_level="Strong"
        )
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128},
        }
        self.collection.create_index("cv_vector", index)


    @staticmethod
    def _is_image_page(page):
        if '/XObject' in page['/Resources']:
            return True
        return False

    @staticmethod
    def _pdf_to_images(pdf_path):
        images = convert_from_path(pdf_path)
        return images

    @staticmethod
    def _extract_text_from_image(image):
        text = pytesseract.image_to_string(image)
        return text

    def _extract_text_from_docx(self,docx_path):
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        return text

    def _extract_text_from_pdf(self,pdf_path):
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                if ResumeParser._is_image_page(page):
                    images = ResumeParser._pdf_to_images(pdf_path)
                    for image in images:
                        text += ResumeParser._extract_text_from_image(image)
                else:
                    text += page.extract_text()
        return text

    @staticmethod
    def _word_embedding(text):
        res = openai.Embedding.create(
            input=text,
            engine="text-embedding-ada-002"
        )
        embed = res.data[0].embedding
        return embed

    def process_resume_files(self, directory_path):
        files = [file for file in os.listdir(directory_path) if file.endswith(('.docx', '.pdf'))]
        threads = []
        
        start = time.time()
        
        for idx, file_name in enumerate(files, 1):
            file_path = os.path.join(directory_path, file_name)
            
            # Create a new thread for each file processing
            t = threading.Thread(target=self.process_file, args=(file_path,file_name))
            t.start()
            threads.append(t)
        
        # Wait for all threads to finish
        for t in threads:
            t.join()
        
        end = time.time()

        seconds = ((end - start) * 10 ** 3) / 1000
        time_in_seconds = round(seconds, 2)

        print("Total Time Taken In Seconds: ", time_in_seconds)
    
    
    def process_file(self, file_path, file_name):
        print(f"Thread Running in {file_path} in Thread-{threading.get_ident()}") 
        if file_name.endswith('.docx'):
            text = self._extract_text_from_docx(file_path)
        elif file_name.endswith('.pdf'):
            text = self._extract_text_from_pdf(file_path) 
        
        embedding = self._word_embedding(text)
        result = self.has_file_already(file_name)
        self._milvusDB(embedding, file_name, text, result)
        print(f"Thread End of {file_path} in Thread-{threading.get_ident()}")       

    def _milvusDB(self, embedding, name_file, text,result):
        if result:
            print("File already exists")
            return
        else:
            obj = {
                "file_name" : name_file,
                "file_content": text,
                "cv_vector": embedding
            }
            # print("Inserting Data In Db")
            insert_result = self.collection.insert([obj])
            self.collection.flush()
            self.collection.load()

    def has_file_already(self,fileName):
        self.collection.load()
        res = self.collection.query(
        expr = f"file_name=='{fileName}'",
        offset = 0,
        limit = 1, 
        output_fields = ["file_name"],
        consistency_level="Strong")

        if res:
            return True
        
        else:
            return False

    def search(self, embed, text):
        self.collection.load()
        vectors_to_search = [embed]
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        result = self.collection.search(vectors_to_search, "cv_vector", search_params, limit=3, output_fields=["file_name","file_content"])
        result = result[0]

        filenames = [item.entity.file_name for item in result]
        filecontents = [item.entity.file_content for item in result]

        for filename, filecontent in zip(filenames, filecontents):
            second_layer(filename, filecontent, text)

