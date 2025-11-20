FROM verlai/verl:app-verl0.4-vllm0.8.5-mcore0.13.0-preview

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=$HTTP_PROXY
ENV http_proxy=${HTTP_PROXY}}
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV https_proxy=${HTTPS_PROXY}}
ENV NO_PROXY=$NO_PROXY
ENV no_proxy=${NO_PROXY}
ENV ALL_PROXY=$ALL_PROXY
ENV HF_HUB_DISABLE_XET=1

WORKDIR /app

RUN apt-get update && \
    # Add NodeSource APT repository for Node.js 20.x
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    # Install Node.js (which includes npm)
    apt-get install -y nodejs

WORKDIR /app
RUN git clone -b multimodal-example https://github.com/konstantin-garbers/agent-lightning
WORKDIR /app/agent-lightning
COPY .env .env
RUN pip install -r examples/multimodal/requirements.txt .[agents]

WORKDIR /app/agent-lightning/examples/multimodal/data
RUN pip install gdown
RUN gdown --id '18T5Meu-bABrKz9zhTT_yRJr2oaaTrskh'
RUN gdown --id '1mZ2NhoIq0fKLZzMrNQhmIfQgXweEZmFm'

WORKDIR /app
RUN git clone https://github.com/volcengine/verl
WORKDIR /app/verl
RUN pip3 install --no-deps -e .

WORKDIR /app
RUN apt-get install -y xvfb
RUN apt-get install -y docker.io

# Dependencies for edgebox (electron)
RUN apt install -y libnss3-dev libatk1.0-0 libatk-bridge2.0-0 libgdk-pixbuf2.0-0 libgtk-3-0