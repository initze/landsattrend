FROM python:3.6-slim

RUN apt-get update && \
      apt-get -y install sudo
RUN apt install -y -qq python3-pip

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install numpy

RUN python3 -m pip install Bottleneck

# Install GDAL dependencies
RUN apt-get install -y libgdal-dev

RUN sudo apt-get install gdal-bin

RUN apt-get install -y build-essential python3-dev python3-setuptools \
                     python3-numpy python3-scipy

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# This will install GDAL
RUN pip3 install GDAL==2.4.4

# Install libspatialindex for Rtree, a ctypes Python wrapper of libspatialindex
RUN apt-get install -y libspatialindex-dev
# create and install the pyincore package


RUN pip3 install pyclowder

RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install --upgrade pip

COPY requirements.txt /home/requirements.txt

COPY test.py /home/test.py

# RUN pip3 install scipy

RUN pip3 install -r /home/requirements.txt

CMD ["python3", "/home/test.py"]