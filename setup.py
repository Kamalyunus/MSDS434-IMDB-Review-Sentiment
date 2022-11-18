from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['transformers==4.1.1', 'pandas==1.1.5', 'scikit-learn==0.24.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application.'
)