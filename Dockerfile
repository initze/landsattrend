FROM python:3.7-slim

RUN apt-get -y update
RUN apt install -y -qq python3-pip

RUN pip3 install numpy==1.18.2
RUN pip3 install numpydoc==0.9.1

# Install GDAL dependencies
RUN apt-get install -y libgdal-dev

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# This will install GDAL
RUN pip3 install GDAL==2.4.2

# Install libspatialindex for Rtree, a ctypes Python wrapper of libspatialindex
RUN apt-get install -y libspatialindex-dev
# create and install the pyincore package

RUN pip3 install scikit-image

RUN pip3 install pyclowder

COPY requirements.txt /home/requirements.txt

RUN pip3 install -r /home/requirements.txt
