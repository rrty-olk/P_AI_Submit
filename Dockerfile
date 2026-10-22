FROM ros:jazzy-ros-base-noble
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-turtlesim python3-venv python3-pip python3-opencv \
    ffmpeg xvfb x11-utils fonts-dejavu-core poppler-utils curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app ROS_DOMAIN_ID=73 QT_X11_NO_MITSHM=1
ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]
