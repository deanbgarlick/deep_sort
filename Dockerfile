FROM tensorflow/tensorflow:1.15.4

WORKDIR /home/microservice

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN git clone https://github.com/deanbgarlick/deep_sort.git

RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libgtk2.0-dev

COPY . .

RUN pip install -r requirements.txt

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]