import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env in user's home directory
load_dotenv(Path.home() / '.env')

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Define the conversation prompt
messages = [{"role": "user", "content": 'Just tell me simply what is "Singapore".'}]


def process_deepseek_chat():
    """Process responses from the non-reasoning model (deepseek-chat)."""
    model_name = "deepseek-chat"
    print(f"\n=== Processing {model_name} ===")
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    last_token_time = None
    content = ""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )
    except Exception as e:
        print(f"API call failed for {model_name}: {e}")
        return

    try:
        for chunk in response:
            delta = chunk.choices[0].delta
            current_time = time.time()

            # Only look for the content field in non-reasoning model
            token = delta.content
            if token not in (None, ""):
                print(token, end="", flush=True)
                content += token
                token_count += 1

                # Record token timings
                if first_token_time is None:
                    first_token_time = current_time
                last_token_time = current_time

    except Exception as e:
        print(f"\nStream error for {model_name}: {e}")
        return

    if token_count == 0:
        print("No tokens received")
        return

    # Calculate metrics
    latency = first_token_time - start_time
    total_duration = last_token_time - first_token_time
    throughput = token_count / total_duration if total_duration > 0 else 0

    print(f"\n\n=== {model_name} Metrics ===")
    print(f"Latency: {latency:.3f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/second")


def process_deepseek_reasoner():
    """Process responses from the reasoning model (deepseek-reasoner)."""
    model_name = "deepseek-reasoner"
    print(f"\n=== Processing {model_name} ===")
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    last_token_time = None
    reasoning_content = ""
    content = ""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )
    except Exception as e:
        print(f"API call failed for {model_name}: {e}")
        return

    try:
        for chunk in response:
            delta = chunk.choices[0].delta
            current_time = time.time()

            # For the reasoning model, first try reasoning_content then fallback to content
            token = None
            if delta.reasoning_content not in (None, ""):
                token = delta.reasoning_content
                reasoning_content += token
            elif delta.content not in (None, ""):
                token = delta.content
                content += token

            if token:
                print(token, end="", flush=True)
                token_count += 1

                if first_token_time is None:
                    first_token_time = current_time
                last_token_time = current_time

    except Exception as e:
        print(f"\nStream error for {model_name}: {e}")
        return

    if token_count == 0:
        print("No tokens received")
        return

    # Calculate metrics
    latency = first_token_time - start_time
    total_duration = last_token_time - first_token_time
    throughput = token_count / total_duration if total_duration > 0 else 0

    print(f"\n\n=== {model_name} Metrics ===")
    print(f"Latency: {latency:.3f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/second")


if __name__ == "__main__":
    process_deepseek_chat()
    process_deepseek_reasoner()
