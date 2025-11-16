# Use Ubuntu base image (Ubuntu in this case)
#FROM ubuntu:18.04
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set timeout for apt-get operations (e.g., 300 seconds = 5 minutes)
RUN echo 'Acquire::http::Timeout "300";' > /etc/apt/apt.conf.d/90timeout

# Install Python 3 and pip
RUN apt-get update

# Install any required utilities (chmod, etc...)
RUN apt-get install -y coreutils
RUN apt-get update && apt-get install -y rename gdb

# Update package lists and install dependencies
RUN apt-get update \
    && apt-get install -y \
        libSDL2-2.0 \
    && rm -rf /var/lib/apt/lists/*

# install Wine
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y wine32 wine64

# Install Python 3 and pip
RUN apt-get install -y python3 python3-pip

# Install OpenJDK Java (you can specify the version you need)
RUN apt-get install -y openjdk-11-jdk

# Set environment variables for Java
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

#FROM python:3.9-buster as python_base
ADD . /nmo_swc
WORKDIR /nmo_swc
RUN mkdir -p /nmo_swc/log
COPY ./ ./

COPY libopenvr_api.so /usr/lib/
COPY libhdf5_serial.so.103 /usr/lib/
COPY libQt6OpenGLWidgets.so.6 /usr/lib/
COPY libQt6Widgets.so.6 /usr/lib/
COPY libQt6Gui.so.6 /usr/lib/
COPY libQt6Core5Compat.so.6 /usr/lib/
COPY libQt6Network.so.6 /usr/lib/
COPY libQt6Xml.so.6 /usr/lib/
COPY libQt6Core.so.6 /usr/lib/
COPY libsz.so.2 /usr/lib/
COPY libQt6OpenGL.so.6 /usr/lib/
COPY libQt6DBus.so.6 /usr/lib/
COPY libicui18n.so.56 /usr/lib/
COPY libicuuc.so.56 /usr/lib/
COPY libicudata.so.56 /usr/lib/
COPY libaec.so.0 /usr/lib/
COPY libEGL.so.1 /usr/lib/
COPY ./plugins/neuron_utilities/neuron_distance/libneuron_dist.so /usr/lib/
RUN ldconfig

RUN pip3 install -r requirements.txt

# Install matplotlib
RUN pip3 install --upgrade pip setuptools
RUN pip3 install Pillow
RUN pip3 install matplotlib 

# Install scipy
RUN pip3 install scipy

# Install scikit
RUN pip3 install scikit-learn

# Install required dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-tk \
    python3-xlib \
    scrot \
    xvfb \
    x11-utils \
    libx11-dev \
    libxtst-dev \
    libpng-dev \
    libgtk-3-dev \
    xserver-xorg-video-dummy

# Install pynput
RUN pip3 install pynput

## COPY install_nano.py /nmo_swc/install_nano.py
RUN python3 install_nano.py

# Install Xvfb
RUN apt-get update && apt-get install -y xvfb

# Set environment variable to use offscreen platform for Qt
ENV QT_QPA_PLATFORM=offscreen

EXPOSE 5000

