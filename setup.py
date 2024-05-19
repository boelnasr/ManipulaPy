#!/usr/bin/env python3


from setuptools import setup, find_packages

setup(
    name='ManipulaPy',
    version='1.0.1',
    author='Mohamed Aboelnar',
    author_email='aboelnasr1997@gmail.com',  # Optional
    description='A package for robotic serial manipulator operations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # if your README is in markdown
    url='https://github.com/boelnasr/ManipulaPy',  # Optional
    packages=find_packages(),
    install_requires=[
    'numpy>=1.19.2',
    'scipy>=1.5.2',
    'urchin>=0.9.0',
    'pybullet>=3.0.6',
    'pycuda>=2021.1'
        ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
