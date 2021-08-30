FROM python:3.6-slim

RUN apt-get update && \
      apt-get -y install sudo
RUN apt install -y -qq python3-pip

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install numpy==1.19.1

RUN python3 -m pip install Bottleneck

# Install GDAL dependencies
RUN apt-get install -y libgdal-dev

RUN sudo apt-get -y install gdal-bin

RUN apt-get install -y build-essential python3-dev python3-setuptools \
                     python3-numpy python3-scipy

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal/bin
ENV C_INCLUDE_PATH=/usr/include/gdal/bin

ENV PATH=/usr/bin/gdal:$PATH

# This will install GDAL
RUN pip3 install GDAL==2.4.4

# Install libspatialindex for Rtree, a ctypes Python wrapper of libspatialindex
RUN apt-get install -y libspatialindex-dev
# create and install the pyincore package


RUN pip3 install pyclowder

RUN pip3 install pika==1.0.0

RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install --upgrade pip

COPY aux_data/dem/DEM.vrt /aux_data/dem/DEM.vrt

COPY aux_data/forestfire/forestfire.vrt /aux_data/forestfire/forestfire.vrt

COPY config /config

COPY landsattrend /landsattrend

COPY models /models

COPY vector /vector

COPY landsattrend_extractor.py /landsattrend_extractor.py

COPY run_lake_analysis.py /run_lake_analysis.py

COPY requirements.txt /requirements.txt

COPY extractor_info.json /extractor_info.json

COPY extractor_info.json /home/extractor_info.json

COPY test.py /test.py

# RUN pip3 install scipy

RUN pip3 install -r /requirements.txt

CMD ["python3", "/landsattrend_extractor.py"]