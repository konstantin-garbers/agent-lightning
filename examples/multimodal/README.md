# Multimodal GUI Agent with Agent Lightning

This is an implementation of a multimodal GUI agent using [Agent Lightning](https://github.com/agent-lightning/agent-lightning). The agent uses [LangGraph](https://github.com/langchain-ai/langgraph) to define its logic and interacts with a sandboxed desktop environment ([EdgeBox](https://github.com/BIGPPWONG/EdgeBox)) via the MCP protocol to complete user tasks.

## 🤖 Agent Workflow

The agent operates in a stateful loop, following an **Observe -> Think -> Act** cycle.


The agent's logic is defined by a LangGraph graph with three primary nodes:
* **`agent_node`**: The "brain" of the agent. A VLM analyzes the original task, the most recent screenshot, and the history of actions to decide what to do next.
* **`tool_node`**: Executes the action (e.g., `desktop_mouse_click`) chosen by the `agent_node`.
* **`screenshot_node`**: Takes a new screenshot of the sandbox after an action is performed, providing the "observation" for the next `agent_node` loop.

When the agent's run is complete (it reaches END) the client sends the original task, the ground truth answer, and the final screenshot to an evaluator VLM (EVALUATOR_MODEL).
The VLM then provides a JSON response with a score of 1 (success) or 0 (failure), which is used as the final reward.


## 🚀 Setup and Installation

Follow these steps to set up the environment.

### 1. Prerequisites

You must have a running instance of **EdgeBox** with GUI tools enabled.
* **Repository**: [https://github.com/BIGPPWONG/EdgeBox](https://github.com/BIGPPWONG/EdgeBox)

### 2. Dataset

Download the required `gui_agent_dataset.parquet` file.

* **Download Link**: [Google Drive](https://drive.google.com/file/d/1ndmi2zUAFuQN9mJENBrXlwzmEtJujgly/view?usp=drive_link)
* **Location**: Based on the project structure, you must create the necessary directories and place the file at this exact relative path from the project root:
    ```bash
    examples/multimodal/data_generation/gui_agent_dataset.parquet
    ```

### 3. Environment Variables

Create a `.env` file in the root of this project and add the following variables.
The provided models should support images and tool usage.

```env
# Endpoint for the agent and evaluator models
OPENAI_API_BASE=[https://api.openai.com/v1](https://api.openai.com/v1)
OPENAI_API_KEY=sk-your-api-key-here

# The VLM to use for the agent's "brain"
AGENT_MODEL=gpt-5-mini

# The VLM to use for evaluating the final result
EVALUATOR_MODEL=gpt-5-mini
```

### 4. Python Dependencies
Install the required Python packages. You can install them individually or save the list below as requirements.txt and run pip install -r requirements.txt.

```
# For data handling
pandas
pyarrow

# Core LangChain/LangGraph
langchain
langgraph
openai

# For connecting to the MCP sandbox
langchain-mcp-adapters

# For the agent framework
agentlightning

# For loading environment variables
python-dotenv
```

## ⚡ How to Run
After completing the setup, you can run the agent:

```
python multimodal_agent.py
```

### Notes & Limitations
API Provider: This agent was tested using OpenAI API endpoints.

Forced Tool Calling: The agent is configured with tool_choice="any". This forces the VLM to call a tool at every step, which is necessary for the agent's loop. Some "OpenAI-compatible" API providers may not support this required tool-use flag and will cause an error.