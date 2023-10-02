FROM python:latest
COPY . /main_v2
WORKDIR /main_v2
RUN pip install -r requirements.txt
CMD python3 -m streamlit run Home.py