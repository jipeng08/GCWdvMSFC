import os
import sys
from setuptools import find_packages, setup
from setuptools.command.install import install
from subprocess import call, check_output
from sys import platform
import numpy
REQUIRED_PACKAGES = ["tensorflow","keras","pykeen","rdflib","pandas","numpy"]
PACKAGE_NAME = 'GCWdvMSFC'
DESCRIPTION = 'Scripts and modules for training and testing deep neural networks that diagnosis of cathode wear in aluminum electrolysis cells using multi-sources feature coupling'
AUTHOR = 'Peng Ji，Gang Yin，Zhuoman Li，Yuehan Yan，Min Wang，Junjie Zhang, Feiya Yan, Peng Zhang, Pengcheng Quan
AUTHOR_EMAIL = 'jipengem@gmail.com'
URL = 'https://github.com/jipeng08/GCWdvMSFC/GCWdvMSFC'
MINIMUM_PYTHON_VERSION = 3, 4
VERSION = '0.0.1'
def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))
check_python_version()
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    url=URL,
    license='Apache License 2.0',
    include_dirs=[numpy.get_include()],
    packages=find_packages()
)
