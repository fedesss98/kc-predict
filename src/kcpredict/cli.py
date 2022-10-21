"""
@author: Federico Amato

Command line Interface to predict Crop Coefficient Kc
"""

import click
from .data.make_data import make_data
from .data.preprocess import preprocess_data
from .models.make_model import make_model
from .prediction.predict import predict

@click.group()
def cli():
    """
    \n\n
    *****************************************************
    \n
    Command Line Interface to predict Crop Coefficient Kc 
    via Machine Learning Models.
    
    Run commands in order first time:\n
    \t- make-data\n
    \t- preprocess-data\n
    \t- make-model\n
    \t- predict\n
    *****************************************************
    """
    return None


@click.group(chain=True)
def make_all():
    """
    Go through all the pipeline
    """
    return None

@click.command()
@click.argument('name', default='Prova')
def say_hello(name):
    click.echo(f'Hello {name}!')
    
    
@click.command()
@click.argument('args', nargs=-1)
def print_args(args):
    print(args)


cli.add_command(make_all)
cli.add_command(make_data)
cli.add_command(preprocess_data)
cli.add_command(make_model)
cli.add_command(predict)

# make_all.add_command(say_hello)
# make_all.add_command(print_args)
# make_all.add_command(make_data)
# make_all.add_command(preprocess_data)
# make_all.add_command(make_model)
# make_all.add_command(predict)




if __name__ == "__main__":
    make_all()