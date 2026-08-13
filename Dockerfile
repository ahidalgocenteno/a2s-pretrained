FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN apt update --fix-missing
RUN apt install build-essential -y
RUN apt install wget -y
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y
RUN apt clean

# MUSTER 
RUN apt install unzip -y
RUN wget https://github.com/TimFelixBeyer/amtevaluation.github.io/raw/refs/heads/main/MUSTER_v231108.zip -O muster.zip
RUN unzip muster.zip -d /MUSTER_tmp
RUN mv /MUSTER_tmp/MUSTER\ 2 /MUSTER\ 2
RUN rm muster.zip && rm -rf /MUSTER_tmp
RUN mv /MUSTER\ 2 /MUSTER
RUN cd /MUSTER && ./compile.sh

# OTHER DEPENDENCIES
RUN pip install --upgrade pip
RUN pip install pybind11
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt