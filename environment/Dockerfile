# hash:sha256:e5977a01201198ffe0c04225a0f15987d074370f8b1c0044f43c9779fc965a58
FROM registry.codeocean.com/codeocean/py-r:python3.12.8-R4.4.2-JupyterLab4.3.5-RStudiorstudio-server-2024.12.1-563-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    matplotlib==3.10.8 \
    numpy==2.4.4 \
    pandas==3.0.2 \
    pathlib==1.0.1 \
    scipy==1.17.1 \
    seaborn==0.13.2
