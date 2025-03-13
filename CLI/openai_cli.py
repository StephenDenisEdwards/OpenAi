#!/usr/bin/env python
import os
import openai
import click

@click.group()
def cli():
    # Retrieve your API key from environment variables
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        click.echo("Error: Please set the OPENAI_API_KEY environment variable.")
        exit(1)

@cli.command()
@click.argument("prompt")
@click.option("--model", default="gpt-3.5-turbo", help="Model to use (default: gpt-3.5-turbo)")
@click.option("--max_tokens", default=150, type=int, help="Max tokens (default: 150)")
def chat(prompt, model, max_tokens):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        click.echo(response.choices[0].message.content.strip())
    except Exception as e:
        click.echo(f"An error occurred: {e}")

@cli.command()
def list_models():
    try:
        models = openai.Model.list()
        for model in models["data"]:
            click.echo(model["id"])
    except Exception as e:
        click.echo(f"An error occurred while listing models: {e}")

if __name__ == '__main__':
    cli()
