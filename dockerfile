FROM python:3.8

COPY app.py /app/app.py
COPY paperTrans.py /app/paperTrans.py

WORKDIR /app

RUN pip install streamlit pandocfilters nltk opencv-python numpy pytesseract pdf2image pygtrans layoutparser mdutils
RUN apt-get update \
  && apt-get -y install tesseract-ocr 

CMD ["streamlit", "run", "app.py"]
