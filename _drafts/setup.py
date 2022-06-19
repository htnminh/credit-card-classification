# DEPRECATED

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(name='MLProject',
      version='1.0',
      packages=find_packages(),
      install_requires=reqs
      )

# \AI-intro-project> pip install .
# \AI-intro-project> pip uninstall MLProject