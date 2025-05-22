from setuptools import setup, find_packages

setup(
    name='timm',
    version='0.3.2',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision'
    ],
)
