from setuptools import setup, find_packages

with open('requirements.txt', "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dotory-space/dotory_fairy_tale_generator",
    version="0.0.1",
    author="DOTORY",
    author_email="developer@dotoryspace.com",
    description="dotory fairy tale generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dotory-space/dotory_fairy_tale_generator.git",
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch',
        'nltk',
        'git+https://github.com/ssut/py-hanspell.git',
    ],
)
