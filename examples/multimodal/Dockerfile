FROM verlai/verl:app-verl0.4-vllm0.8.5-mcore0.13.0-preview

WORKDIR /app/edgebox
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    # Install Node.js (which includes npm)
    apt-get install -y nodejs && \
    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/konstantin-garbers/EdgeBox
RUN npm install

WORKDIR /app

COPY pyproject.toml .
COPY examples/multimodal/requirements.txt .
COPY README.md README.md
RUN pip install -r requirements.txt .[agents]

COPY . .

RUN chmod +x ./examples/multimodal/entrypoint.sh

RUN apt-get install -y xvfb
RUN apt-get install -y docker.io