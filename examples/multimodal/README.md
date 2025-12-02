# Multimodal GUI Agent with Agent Lightning

This is an implementation of a multimodal GUI agent using [Agent Lightning](https://github.com/agent-lightning/agent-lightning). The agent uses [LangGraph](https://github.com/langchain-ai/langgraph) to define its logic and interacts with a sandboxed desktop environment ([EdgeBox](https://github.com/BIGPPWONG/EdgeBox)) via the MCP protocol to complete user tasks.

The agent is trained using reinforcement learning with [VERL](https://github.com/microsoft/verl) and runs inference using [vLLM](https://github.com/vllm-project/vllm) for efficient VLM serving.

## 🤖 Agent Workflow

The agent operates in a stateful loop, following an **Observe -> Think -> Act** cycle.

The agent's logic is defined by a LangGraph graph with three primary nodes:
* **`agent_node`**: The "brain" of the agent. A Vision Language Model (VLM) analyzes the original task, the most recent screenshot, and the history of actions to decide what to do next.
* **`tool_node`**: Executes the action (e.g., `desktop_mouse_click`) chosen by the `agent_node`.
* **`screenshot_node`**: Takes a new screenshot of the sandbox after an action is performed, providing the "observation" for the next `agent_node` loop.

When the agent's run is complete (it reaches END), the client sends the original task, the ground truth answer, and the final screenshot to an evaluator VLM. The VLM then provides a JSON response with a score of 1 (success) or 0 (failure), which is used as the final reward for reinforcement learning.

## 🚀 Setup and Installation

### Prerequisites

- **NVIDIA GPU** with CUDA support (required for vLLM)
- **Docker** with GPU support (nvidia-docker2)
- **EdgeBox** sandbox environment (see EdgeBox Setup section below)

### Architecture Overview

This setup uses:
- **vLLM 0.10.2**: For efficient VLM inference serving
- **VERL 0.6.0**: For reinforcement learning training
- **EdgeBox**: Sandboxed desktop environment for GUI interactions
- **Agent Lightning**: Framework for agent training and deployment

## 🐳 Docker Setup (Recommended)

The recommended way to run this agent is using Docker, which handles all dependencies including CUDA, vLLM, and VERL.

### 1. Build the Docker Image

Build the Docker image with the necessary dependencies:

```bash
docker build \
  --network=host \
  -t multimodal:0 \
  -f examples/multimodal/Dockerfile \
  --build-arg HTTP_PROXY="http://127.0.0.1:7890" \
  --build-arg HTTPS_PROXY="http://127.0.0.1:7890" \
  --build-arg ALL_PROXY="http://127.0.0.1:7890" \
  --build-arg NO_PROXY="localhost,127.0.0.1" \
  --build-arg no_proxy="localhost,127.0.0.1" \
  .
```

**Note**: Adjust proxy settings (`HTTP_PROXY`, `HTTPS_PROXY`, etc.) according to your network configuration. If you don't need a proxy, you can omit these build arguments.

### 2. Run the Docker Container

Run the container with GPU support:

```bash
docker run \
  -it \
  --rm \
  --name multimodal \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network=host \
  --shm-size="10g" \
  --add-host=host.docker.internal:host-gateway \
  -e HTTP_PROXY="http://host.docker.internal:7890" \
  -e http_proxy="http://host.docker.internal:7890" \
  -e HTTPS_PROXY="http://host.docker.internal:7890" \
  -e https_proxy="http://host.docker.internal:7890" \
  -e NO_PROXY="localhost,127.0.0.1,host.docker.internal" \
  -e no_proxy="localhost,127.0.0.1,host.docker.internal" \
  -e ALL_PROXY="http://host.docker.internal:7890" \
  -e HF_HUB_DISABLE_XET=1 \
  -e WANDB_MODE=disabled \
  -v /var/run/docker.sock:/var/run/docker.sock \
  multimodal:0
```

**Important**: Adjust proxy environment variables according to your setup. The container needs access to the host's Docker socket (`/var/run/docker.sock`) to manage EdgeBox sandbox containers.

### 3. Enter the Container

Once the container is running, you can access it and execute the python script inside it:

```bash
docker exec -it multimodal /bin/bash
```

### 4. Dataset

The dataset files are automatically downloaded during the Docker build process using `gdown`. The following files are downloaded to `examples/multimodal/data/`:

- **Train set**: [train_set.parquet](https://drive.google.com/file/d/18T5Meu-bABrKz9zhTT_yRJr2oaaTrskh/view?usp=sharing)
- **Validation set**: [validation_set.parquet](https://drive.google.com/file/d/1mZ2NhoIq0fKLZzMrNQhmIfQgXweEZmFm/view?usp=sharing)

If you need to download them manually:

```bash
cd /app/agent-lightning/examples/multimodal/data
pip install gdown
gdown https://drive.google.com/uc?id=18T5Meu-bABrKz9zhTT_yRJr2oaaTrskh  # train_set.parquet
gdown https://drive.google.com/uc?id=1mZ2NhoIq0fKLZzMrNQhmIfQgXweEZmFm  # validation_set.parquet
```

### 5. Environment Variables

When using VERL for training, you typically don't need to set `OPENAI_API_BASE` or `OPENAI_API_KEY` as VERL manages the vLLM backend internally.

However, if you're debugging the agent locally without VERL (e.g., using `python -m examples.multimodal.multimodal_agent`), create a `.env` file in the project root with:

```env
# Endpoint for the VLM serving (vLLM typically runs on localhost:8000)
# Only needed when debugging without VERL
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=dummy-key  # vLLM doesn't require a real key

# The VLM model to use for the agent's "brain"
# This should match a model loaded in your vLLM server
AGENT_MODEL=Qwen/Qwen2.5-VL-3B-Instruct

# The VLM model to use for evaluating the final result
EVALUATOR_MODEL=Qwen/Qwen2.5-VL-3B-Instruct

# Optional: separate evaluator endpoint/key (falls back to the values above)
EVALUATION_LLM_API_BASE=http://localhost:8001/v1
EVALUATION_LLM_API_KEY=another-dummy-key
```

If `EVALUATION_LLM_API_BASE` or `EVALUATION_LLM_API_KEY` are not set, the agent reuses `OPENAI_API_BASE`/`OPENAI_API_KEY`.

## 🖥️ EdgeBox Setup

EdgeBox is the sandboxed desktop environment that the multimodal agent uses to perform GUI actions. You need to run EdgeBox separately before training the agent.

### 1. Clone EdgeBox Repository

```bash
git clone https://github.com/konstantin-garbers/EdgeBox.git
cd EdgeBox
```

### 2. Install Dependencies

```bash
pnpm install
```

### 3. Run EdgeBox

**In a server environment** (with virtual display):

```bash
xvfb-run -a pnpm start -- --no-sandbox
```

**On a development environment** (with display):

```bash
pnpm start
```

EdgeBox will start an MCP server that the multimodal agent connects to. By default, it runs on `http://localhost:8888/mcp`.

### Optional: Build Custom Sandbox Image

If you need to customize the EdgeBox sandbox image:

```bash
docker buildx build \
  --network=host \
  --build-arg HTTP_PROXY="http://127.0.0.1:10808" \
  --build-arg HTTPS_PROXY="http://127.0.0.1:10808" \
  --build-arg NO_PROXY="localhost,127.0.0.1,::1" \
  --platform linux/amd64 \
  --tag wurstm162/e2b-sandbox:latest \
  --load .
```

## ⚡ Running the Agent

### Training

Train the multimodal agent using the provided training script:

```bash
# Fast training (for CI/testing)
python examples/multimodal/train_multimodal_agent.py fast

# Standard Qwen model training
python examples/multimodal/train_multimodal_agent.py qwen
```

The training script supports additional options:

```bash
python examples/multimodal/train_multimodal_agent.py qwen \
  --active-agent "custom_agent_name" \
  --save-screenshot \
  --screenshot-output-folder ./screenshots
```

### Debug Mode

Run the agent in debug mode for development:

```bash
python -m examples/multimodal/multimodal_agent.py 
```

This will run a single rollout with debug logging enabled and save screenshots automatically.

## 📋 Local Setup (Without Docker)

If you prefer to set up the environment locally without Docker:

### Requirements

- Python 3.10+
- CUDA 12.4+ compatible GPU
- verl >= 0.6.0
- vllm >= 0.10.2

### Installation Steps

1. **Install vLLM** (this will install the correct PyTorch version):

```bash
pip install vllm==0.10.2
```

2. **Install Flash Attention**:

```bash
pip install flash-attn --no-build-isolation
```

3. **Install VERL**:

```bash
pip install verl==0.6.0
```

4. **Install project dependencies**:

```bash
cd examples/multimodal
pip install -r requirements.txt
pip install -e ../..[agents]
```

5. **Download datasets** (see Dataset section above)

6. **Set up environment variables** (see Environment Variables section above)

## 🔧 Configuration

### Training Configurations

The training script supports two configurations:

1. **`fast`**: Lightweight configuration for CI/testing
   - Uses `Qwen/Qwen2.5-Coder-0.5B-Instruct` model
   - 1 epoch, 1 training step
   - Optimized for quick validation

2. **`qwen`**: Standard configuration
   - Uses `Qwen/Qwen2.5-VL-3B-Instruct` model
   - 2 epochs
   - Full training configuration

### Model Configuration

The agent uses vLLM for model serving. Ensure your vLLM server is running and configured to serve the models specified in your `.env` file. The agent connects to vLLM via the OpenAI-compatible API endpoint.

## 📝 Notes & Limitations

- **GPU Requirements**: vLLM requires NVIDIA GPUs with CUDA support. The Docker setup includes CUDA 12.4.

- **EdgeBox Dependency**: EdgeBox must be running before training starts. The agent connects to EdgeBox's MCP server at `http://localhost:8888/mcp`.

- **Tool Calling**: The agent is configured with `tool_choice="any"`, which forces the VLM to call a tool at every step. This is necessary for the agent's loop and is supported by vLLM's OpenAI-compatible API.

- **Screenshot Saving**: When `debug=True` or `--save-screenshot` is used, screenshots are automatically saved to `./screenshots/` organized by rollout ID.
