FROM python:3.9
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "footwear_app.py"]

# ENTRYPOINT ["streamlit", "run", "footwear_app.py", "--server.port=8501", "--server.address=0.0.0.0"]