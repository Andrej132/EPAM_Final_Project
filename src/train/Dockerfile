FROM python:3.9-slim

WORKDIR /app

COPY ../../src ./src
COPY ../../data ./data
COPY ../../outputs ./outputs

COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt wordnet stopwords punkt_tab

ENV PYTHONPATH=/app

CMD ["python", "src/train/train.py"]