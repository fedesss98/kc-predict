import click
from .data.make_data import get_features

@click.command()
def cli():
    """Example script."""
    click.echo('Hello World !!')

if __name__ == "__main__":
    print('Ok')