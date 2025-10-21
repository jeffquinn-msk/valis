FROM bitnami/pytorch:latest

USER root

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/.cache/pip pip install ipython ipdb memray
RUN apt-get update && apt-get install --no-install-recommends -y \
	libglib2.0-dev \
	glib-2.0-dev \
	libexpat1-dev \
	libexpat-dev \
	librsvg2-2 \
	librsvg2-common \
	librsvg2-dev \
	libpng-dev \
	libjpeg-turbo8-dev \
	libopenjp2-7-dev \
	libtiff-dev \
	libexif-dev \
	liblcms2-dev \
	libheif-dev \
	liborc-dev \
  	libgirepository1.0-dev \
	libopenslide-dev \
	librsvg2-dev \
    libvips-dev

RUN mkdir -p /app
COPY src/ /app/src/

COPY pyproject.toml setup.py LICENSE.txt README.rst /app/

RUN --mount=type=cache,target=/.cache/pip cd /app && pip install '.[dev,test]'

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Download pytorch model weights
COPY ./docker/docker_download_weights.py docker_download_weights.py
RUN python3 docker_download_weights.py


