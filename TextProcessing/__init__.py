import os

try:
    import pandas
except ImportError as e:
    os.system('python -m pip install pandas')

try:
    import nltk
except ImportError as e:
    os.system('python -m pip install nltk')

try:
    import numpy
except ImportError as e:
    os.system('python -m pip install numpy')

try:
    import wordcloud
except ImportError as e:
    os.system('python -m pip install wordcloud')

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    os.system('python -m pip install matplotlib')

try:
    import beautifulsoup4
except ImportError as e:
    os.system('python -m pip install beautifulsoup4')