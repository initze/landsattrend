FROM python:3.8-slim

RUN apt-get -y update
RUN apt install -y -qq python3-pip

RUN python3 -m pip install --upgrade pip

RUN pip3 install Bottleneck==1.3.2

RUN pip3 install numpy==1.22.0

# Install GDAL dependencies
RUN apt-get install gdal-bin -y
RUN apt-get install -y libgdal-dev -y

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal/bin
ENV C_INCLUDE_PATH=/usr/include/gdal/bin

ENV PATH=/usr/bin/gdal:$PATH

RUN gdal-config --version

## This will install GDAL
RUN pip3 install GDAL==3.2.2
#
# Install libspatialindex for Rtree, a ctypes Python wrapper of libspatialindex
RUN apt-get install -y libspatialindex-dev
# create and install the pyincore package


RUN pip3 install pyclowder

COPY requirements.txt /home/requirements.txt

RUN pip3 install -r /home/requirements.txt
