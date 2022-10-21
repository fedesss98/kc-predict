import click
from .data.make_data import make_data
from .data.preprocess import preprocess_data
from .models.make_model import make_model

@click.group()
def cli():
    """Command Line Interface to predict Crop Coefficient Kc via Machine Learning Models."""
    return None

cli.add_command(make_data)
cli.add_command(preprocess_data)
cli.add_command(make_model)

if __name__ == "__main__":
    cli()