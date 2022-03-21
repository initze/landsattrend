FROM ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version

RUN conda clean -a

RUN echo $CONDA_PREFIX

COPY extractor_info.json .


COPY aux_data ./aux_data

COPY config ./config

COPY landsattrend ./landsattrend

COPY models ./models

COPY environment_py38_v2_extractor.yml environment_py38_v2_extractor.yml

COPY extractor_info.json extractor_info.json

COPY lake_analysis.py lake_analysis.py

COPY test.py test.py

COPY lake_analysis_extractor.py lake_analysis_extractor.py

COPY requirements.txt requirements.txt

COPY setup.py setup.py

RUN ls

RUN conda install -c conda-forge mamba

RUN mamba env create -f environment_py38_v2_extractor.yml

SHELL ["conda", "run", "-n", "landsattrend2", "/bin/bash", "-c"]

RUN python -m pip install --ignore-installed pyclowder


CMD ["conda", "run", "--no-capture-output", "-n", "landsattrend2", "python","-u", "/lake_analysis_extractor.py"]