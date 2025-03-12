#!/usr/bin/env python
import openai
import argparse
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Simple CLI for interacting with OpenAI's API (latest version)")
    parser.add_argument("prompt", type=str, help="The prompt to send to OpenAI")
    parser.add_argument("--model", type=str, default="gpt-4", help="The model to use (default: gpt-4)")
    parser.add_argument("--max_tokens", type=int, default=150, help="Maximum number of tokens to generate (default: 150)")
    args = parser.parse_args()
    
    # Retrieve API key from environment
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        exit(1)
    
    try:
        # Make a request to the latest version of the ChatCompletion API
        response = openai.ChatCompletion.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            temperature=0.7
        )
        print(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()