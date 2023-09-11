"""A setuptools based setup module for TTSE_Project.
See:
https://github.com/mayank05942/TTSE_Project

"""
# Taken from sciope

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='TTSE_Project',

    version='0.1',

    description='TTSE_Project',
    long_description_content_type='text/markdown',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/mayank05942/TTSE_Project',

    # Author details
    author='Mayank Nautiyal',
    author_email='mayanknautiyal88@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Your preferred license here',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',  # or your supported version
    ],

    keywords='keywords related to TTSE_Project',

    packages=find_packages(),

    install_requires=[
        'numpy>=1.16.5',
        'scipy',
        'scikit-learn',
        'gillespy2',
        'tsfresh==0.20.1',
        'ipywidgets',
        'dask',
        'distributed',
        'matplotlib',
        'umap-learn'],

    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },

    
)
