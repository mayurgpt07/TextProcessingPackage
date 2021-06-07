from setuptools import setup, find_packages

setup(name='TextPreProc',
version='1.0',
description='The package is created to simplify a users effort of text clearning and exploration. It allows user to clean the data and do some basic analysis like N-gram WordCouds and Topic Modelling ',
url='https://github.com/mayurgpt07/TextProcessingPackage/',
author='auth',
author_email='mayurgpt07@gmail.com',
license='MIT',
packages=find_packages(),
install_requires = [
    'pandas',
    'nltk',
    'numpy',
    'matplotlib',
    'wordcloud',
    'beautifulsoup4',
    'gensim',
    'sklearn',
    'seaborn'
],
zip_safe=False)