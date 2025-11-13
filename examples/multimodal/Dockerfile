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

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    # Install Node.js (which includes npm)
    apt-get install -y nodejs && \

RUN git clone https://github.com/konstantin-garbers/EdgeBox
WORKDIR /app/EdgeBox
RUN npm install

WORKDIR /app
RUN git clone -b multimodal-example https://github.com/konstantin-garbers/agent-lightning
WORKDIR /app/agent-lightning
COPY .env .env
RUN pip install -r examples/multimodal/requirements.txt .[agents]

RUN apt-get install -y xvfb
RUN apt-get install -y docker.io