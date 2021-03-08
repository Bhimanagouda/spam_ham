FROM ubuntu:18.04
RUN apt-get -y update && apt-get install -y --no-install-recommends wget python3 nginx ca-certificates python3-distutils
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py 
#ENV PATH="/home/ubuntu:${PATH}"
COPY . src/
RUN /bin/bash -c "cd src && pip install -r requirements.txt"
COPY . /home/ubuntu/spam_ham/
WORKDIR /home/ubuntu/spam_ham/
CMD python3 app.py
EXPOSE 80