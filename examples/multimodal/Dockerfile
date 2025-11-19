FROM verlai/verl:app-verl0.4-vllm0.8.5-mcore0.13.0-preview

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG no_proxy

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV ALL_PROXY=$ALL_PROXY
ENV NO_PROXY=$NO_PROXY
ENV no_proxy=$no_proxy

WORKDIR /app

RUN apt-get update && \
    # Add NodeSource APT repository for Node.js 20.x
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    # Install Node.js (which includes npm)
    apt-get install -y nodejs

WORKDIR /app
RUN git clone -b multimodal-example https://github.com/konstantin-garbers/agent-lightning
COPY .env .env
RUN pip install -r examples/multimodal/requirements.txt .[agents]

WORKDIR /app/examples/multimodal/data_generation
RUN pip install gdown
RUN gdown --id '1ndmi2zUAFuQN9mJENBrXlwzmEtJujgly' -O gui_agent_dataset.parquet

WORKDIR /app
RUN apt-get install -y xvfb
RUN apt-get install -y docker.io

# Dependencies for edgebox (electron)
RUN apt install -y libnss3-dev libatk1.0-0 libatk-bridge2.0-0 libgdk-pixbuf2.0-0 libgtk-3-0