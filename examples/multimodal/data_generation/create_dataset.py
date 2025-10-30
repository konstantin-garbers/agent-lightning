import os
import json
import time  # Added for rate limiting
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import pyarrow  # Required by pandas for the Parquet engine

# --- Configuration ---
OUTPUT_FILE = "gui_agent_dataset.parquet"
OPENAI_MODEL = "Qwen/Qwen2.5-32B-Instruct"
NUM_TASKS_TO_GENERATE = 200  # Total number of tasks to generate
# ---------------------


def get_system_prompt():
    """
    Creates the system prompt to instruct the LLM on how to generate
    a single task-answer pair.
    """
    return """
    You are an expert in creating evaluation datasets for GUI agents.
    The agent operates in a sandbox with access to two applications:
    1. Google Chrome (a web browser)
    2. Visual Studio Code (VS Code, a text editor)

    Your job is to generate *one* simple, concrete task.
    The task must be solvable using *only* GUI operations (mouse clicks, typing)
    within Chrome or VS Code.

    Provide two fields in a JSON object:
    1. "task": A short, imperative command for the agent.
       (e.g., 'Open Chrome and search for the capital of France.')
    2. "answer": A detailed description of the *expected final state* of the GUI.
       This description will be given to an LLM judge to verify success.
       It must be specific and verifiable by looking at the screen.

    Here are some examples of the format:
    ---
    Example 1:
    {
      "task": "Use Chrome to find today's weather in London.",
      "answer": "The Google Chrome window is active. A Google search for 'weather in London'
                 has been performed, and the current weather forecast (e.g., temperature,
                 conditions like 'Cloudy') is clearly visible on the screen."
    }
    ---
    Example 2:
    {
      "task": "Create a new file in VS Code, write 'Hello World' in it, and
               save it as 'test.txt' in the home directory.",
      "answer": "The VS Code window is active. A new tab named 'test.txt' is open
                 and its content is the exact text 'Hello World'. The file 'test.txt'
                 is visible in the VS Code file explorer sidebar, confirming it is saved."
    }
    ---
    Example 3:
    {
      "task": "Open Chrome, go to wikipedia.org, and search for 'Python programming language'.",
      "answer": "The Google Chrome window is active and the address bar shows 'wikipedia.org'.
                 The main content area shows the Wikipedia article for
                 'Python (programming language)'."
    }
    ---

    Generate *one* new, unique task.
    Return *only* a single valid JSON object, with no other text,
    headers, lists, or explanations.
    """


def generate_single_task(client: OpenAI) -> dict[str, str] | None:
    """
    Calls the OpenAI API to generate one new task-answer pair.
    Returns a single dictionary or None on failure.
    """
    print(f"Attempting to generate 1 new task using {OPENAI_MODEL}...")
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": "Generate one new task."}
            ],
            # Use JSON mode for reliable output
            response_format={"type": "json_object"}
        )

        response_content: str = str(completion.choices[0].message.content) if completion.choices else ""
        
        # The prompt asks for a single object.
        raw_data = json.loads(response_content)
        
        task_obj = None

        # Case 1: Model returns a single object: {"task": ...}
        if isinstance(raw_data, dict) and "task" in raw_data:
            task_obj = raw_data
        # Case 2: Model wraps it in a dict: {"task_1": {"task": ...}} or {"task": {...}}
        elif isinstance(raw_data, dict) and len(raw_data) > 0:
            first_value = next(iter(raw_data.values()))
            if isinstance(first_value, dict) and "task" in first_value:
                task_obj = first_value
        # Case 3: Model *still* returns a list with one item: [{"task": ...}]
        elif isinstance(raw_data, list) and len(raw_data) > 0:
             task_obj = raw_data[0]
        
        # Validate the final object
        if isinstance(task_obj, dict) and "task" in task_obj and "answer" in task_obj:
            print(f"  > Success: {task_obj['task'][:60]}...")
            return task_obj
        else:
            print(f"Warning: Skipping invalid/unexpected item format: {task_obj}")
            return None

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON response from OpenAI: {e}")
        print(f"Raw response: {response_content}")
        return None
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None


def save_or_append_data(new_data_list: list[dict], filename: str):
    """
    Saves new data to a Parquet file.
    Appends to the file if it already exists.
    """
    if not new_data_list:
        print("No new data to save.")
        return

    # Convert the new list of dicts to a DataFrame
    new_df = pd.DataFrame(new_data_list, columns=["task", "answer"])

    if os.path.exists(filename):
        print(f"File '{filename}' exists. Appending new data...")
        try:
            # Read the existing data
            existing_df = pd.read_parquet(filename, engine='pyarrow')
            
            # Concatenate old and new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Save the combined DataFrame
            combined_df.to_parquet(filename, index=False, engine='pyarrow')
            
            print(f"Successfully appended {len(new_df)} rows.")
            print(f"Total rows in '{filename}': {len(combined_df)}")
            
        except Exception as e:
            print(f"Error reading or writing existing Parquet file: {e}")
    else:
        print(f"File '{filename}' not found. Creating new file...")
        try:
            # Save the new DataFrame as a new file
            new_df.to_parquet(filename, index=False, engine='pyarrow')
            print(f"Successfully created '{filename}' with {len(new_df)} rows.")
        except Exception as e:
            print(f"Error writing new Parquet file: {e}")


def main():
    load_dotenv()
    
    if "OPENAI_API_KEY" not in os.environ:
        print("="*50)
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key before running the script:")
        print("  export OPENAI_API_KEY='your_key_here'")
        print("="*50)
        return

    try:
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_API_BASE"), # Use .get for optional base_url
        )
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return

    all_new_tasks = []
    print(f"--- Starting Generation of {NUM_TASKS_TO_GENERATE} Tasks ---")

    for i in range(NUM_TASKS_TO_GENERATE):
        print(f"\n--- Generating Task {i + 1}/{NUM_TASKS_TO_GENERATE} ---")
        
        # 1. Generate a single task
        new_task = generate_single_task(client)

        if new_task:
            all_new_tasks.append(new_task)
        else:
            print(f"Failed to generate task {i + 1}. Skipping.")
        
        # Be polite to the API to avoid rate limits
        time.sleep(1)

    print("\n--- Generation Complete ---")

    # 2. Save or append all new tasks to the Parquet file
    if all_new_tasks:
        print(f"Successfully generated {len(all_new_tasks)} total tasks.")
        save_or_append_data(all_new_tasks, OUTPUT_FILE)
    else:
        print("Task generation failed. No data to save.")

if __name__ == "__main__":
    main()