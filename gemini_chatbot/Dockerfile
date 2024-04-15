FROM python:3.8

RUN pip install gradio>=4.26.0
RUN pip install google-cloud-aiplatform==1.47.0 google-cloud-logging

COPY ./src /src


WORKDIR /src

EXPOSE 7860

CMD ["python", "chatbot.py"]