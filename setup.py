from setuptools import setup, find_packages

setup(name='TextPreProc',
version='0.6',
description='Text preprocessing made easy in python',
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
    'sklearn'
],
zip_safe=False)