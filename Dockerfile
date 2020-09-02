FROM python:3.7-slim

RUN apt-get -y update
RUN apt install -y -qq python3-pip

RUN python3 -m pip install --upgrade pip

RUN pip3 install Bottleneck==1.2.1

RUN pip3 install numpy==1.17.3

# Install GDAL dependencies
RUN apt-get install -y libgdal-dev

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# This will install GDAL
RUN pip3 install GDAL==2.4.3

# Install libspatialindex for Rtree, a ctypes Python wrapper of libspatialindex
RUN apt-get install -y libspatialindex-dev
# create and install the pyincore package


RUN pip3 install pyclowder

COPY requirements.txt /home/requirements.txt

RUN pip3 install -r /home/requirements.txt
