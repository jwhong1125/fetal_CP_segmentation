FROM ubuntu:18.04
MAINTAINER Jinwoo Hong "jwhong1125@gmail.com"

ENV APPROOT="/usr/src/fetal_cp_seg"
COPY ["fetal_cp_seg", "${APPROOT}"]
COPY ["requirements_dev.txt", "${APPROOT}"]

WORKDIR $APPROOT

RUN apt update
RUN apt-get install -y python3 python3-pip


RUN pip3 install --upgrade pip
RUN pip3 install -r requirements_dev.txt

CMD ["python3" "fetal_CP_seg.py", "--help"]
