# 1. Base Image
FROM verlai/verl:app-verl0.4-vllm0.8.5-mcore0.13.0-preview

WORKDIR /app

COPY pyproject.toml .
COPY examples/multimodal/requirements.txt .
RUN pip install -r examples/multimodal/requirements.txt .[agents]


RUN apt-get update && \
    apt-get install -y edgebox_1.0.0_amd64.deb && \
    rm edgebox_1.0.0_amd64.deb

COPY . .

RUN chmod +x ./examples/multimodal/entrypoint.sh

CMD ["./examples/multimodal/entrypoint.sh"]