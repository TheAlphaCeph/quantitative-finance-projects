"""
Setup script for momentum_strategy_project
"""

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='momentum_strategy',
    version='1.0.0',
    author='Abhay Kanwar',
    author_email='abhaykanwar@uchicago.edu',
    description='Frog-in-the-Pan momentum strategy with NLP sentiment analysis',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.9'
)
