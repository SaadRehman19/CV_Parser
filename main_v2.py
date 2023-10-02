from Embedding import ResumeParser

resume_parser = ResumeParser()
directory_path = r"/home/gaditek/CV_To_WordEmbedding/CV_File"
resume_parser.process_resume_files(directory_path)