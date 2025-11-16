# Use Ubuntu base image (Ubuntu in this case)
#FROM ubuntu:18.04
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set timeout for apt-get operations (e.g., 300 seconds = 5 minutes)
RUN echo 'Acquire::http::Timeout "300";' > /etc/apt/apt.conf.d/90timeout

# Install sudo
RUN apt-get update && apt-get install -y sudo

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

# ------------------ Install Tomcat ------------------
RUN apt-get update && apt-get install -y curl \
    && mkdir -p /opt/tomcat
WORKDIR /opt/tomcat

# Download Tomcat 9 latest stable
RUN curl -O https://archive.apache.org/dist/tomcat/tomcat-9/v9.0.84/bin/apache-tomcat-9.0.84.tar.gz \
    && tar xzvf apache-tomcat-9.0.84.tar.gz --strip-components=1 \
    && rm apache-tomcat-9.0.84.tar.gz

# Fix permissions and directories
RUN chmod +x /opt/tomcat/bin/*.sh \
    && mkdir -p /opt/tomcat/temp /opt/tomcat/logs /opt/tomcat/webapps \
    && touch /opt/tomcat/logs/catalina.out \
    && chown -R root:root /opt/tomcat \
    && chmod -R 755 /opt/tomcat

# Create Tomcat lib directory if it doesn't exist
RUN mkdir -p /opt/tomcat/lib \
    && chmod -R 755 /opt/tomcat/lib

# Download JAXB runtime jars into Tomcat lib
RUN curl -L -o /opt/tomcat/lib/jaxb-api-2.3.1.jar https://repo1.maven.org/maven2/javax/xml/bind/jaxb-api/2.3.1/jaxb-api-2.3.1.jar \
 && curl -L -o /opt/tomcat/lib/jaxb-runtime-2.3.5.jar https://repo1.maven.org/maven2/org/glassfish/jaxb/jaxb-runtime/2.3.5/jaxb-runtime-2.3.5.jar \
 && curl -L -o /opt/tomcat/lib/jaxb-core-2.3.0.1.jar https://repo1.maven.org/maven2/com/sun/xml/bind/jaxb-core/2.3.0.1/jaxb-core-2.3.0.1.jar \
 && curl -L -o /opt/tomcat/lib/jaxb-impl-2.3.5.jar https://repo1.maven.org/maven2/org/glassfish/jaxb/jaxb-impl/2.3.5/jaxb-impl-2.3.5.jar

# Copy your WAR and SWC folder
COPY swc.war /opt/tomcat/webapps/
COPY swc/ /opt/tomcat/webapps/swc/

# Ensure webapps folder is writable
RUN chmod -R 755 /opt/tomcat/webapps

# ------------------ Node.js --------------------------
RUN apt-get remove -y nodejs npm || true \
    && apt-get purge -y nodejs npm || true \
    && apt-get autoremove -y

RUN apt-get update && apt-get install -y curl gnupg build-essential

# Install Node 20 from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

RUN node -v && npm -v

RUN apt-get update && apt-get install -y xvfb \
    gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 \
    libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 \
    libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 \
    libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 \
    libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 \
    libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 \
    libxtst6 ca-certificates fonts-liberation xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Puppeteer
WORKDIR /nmo_swc
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=false
RUN npx @puppeteer/browsers install chrome@stable

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libgl1-mesa-dri libosmesa6 \
    libx11-6 libxext6 libxrender1 libxi6 libxrandr2 \
    libxinerama1 libxfixes3 libxcomposite1 libxcursor1 \
    libxdamage1 libxss1 libxtst6

## COPY install_nano.py /nmo_swc/install_nano.py
RUN python3 install_nano.py

# Set environment variable to use offscreen platform for Qt
ENV QT_QPA_PLATFORM=offscreen

EXPOSE 5000 8080 
