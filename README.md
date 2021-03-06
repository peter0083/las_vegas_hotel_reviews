# las_vegas_hotel_reviews
a short machine learning exercise for my Las Vegas trip in June 2018

by Peter Lin

### Summary

This repo contains a short analysis of quantitative and categorical features from online reviews from 21 hotels located in Las Vegas Strip, extracted from TripAdvisor. The original [data](http://archive.ics.uci.edu/ml/machine-learning-databases/00397/LasVegasTripAdvisorReviews-Dataset.csv) is downloaded from the [Machine Learning Repository of UC Irvine](http://archive.ics.uci.edu/ml/datasets/Las+Vegas+Strip) into the [ipython notebook](https://github.com/peter0083/las_vegas_hotel_reviews/blob/master/src/vegas_trip_analysis.ipynb). The dataset contains 504 records and tuned features from
24 per hotel (two per each month, randomly selected), regarding the year of 2015. The CSV contains a header, with the names of the columns corresponding to the features marked Exploratory data analysis and machine learning models are built in hope to identify key features that result in high or low hotel scores. The analysis can be reproduced by cloning this repo to your local machine and executing the code cells in the ipython notebook. The dependencies of this ipython notebook is listed in this readme and also at the end of the notebook. This ipython notebook can also be found in the 'src' folder of this repo.

**citation of original data**

Moro, S., Rita, P., & Coelho, J. (2017). Stripping customers' feedback on hotels through data mining: The case of Las Vegas Strip. Tourism Management Perspectives, 23, 41-52.

### License

Please see the license information [here](https://github.com/peter0083/las_vegas_hotel_reviews/blob/master/LICENSE).

### Next Step

Try NLP on the actual reviews

### Dependencies

Python:

```bash
alabaster==0.7.10
anaconda-clean==1.0
anaconda-client==1.6.3
anaconda-navigator==1.6.2
anaconda-project==0.6.0
appnope==0.1.0
appscript==1.0.1
argcomplete==1.0.0
asn1crypto==0.22.0
astroid==1.4.9
astropy==1.3.2
attrs==16.3.0
Babel==2.4.0
backports.shutil-get-terminal-size==1.0.0
beautifulsoup4==4.6.0
bitarray==0.8.1
blaze==0.10.1
bleach==1.5.0
bokeh==0.12.5
boto==2.46.1
Bottleneck==1.2.1
bs4==0.0.1
cffi==1.10.0
chardet==3.0.3
chest==0.2.3
click==6.7
cloudpickle==0.2.2
clyent==1.2.2
colorama==0.3.9
conda==4.3.25
conda-build==2.0.2
configobj==5.0.6
constantly==15.1.0
contextlib2==0.5.5
cryptography==1.8.1
cssselect==1.0.1
cycler==0.10.0
Cython==0.25.2
cytoolz==0.8.2
dask==0.14.3
datashape==0.5.4
decorator==4.0.11
dill==0.2.5
distributed==1.16.3
docutils==0.13.1
dynd==0.7.3.dev1
entrypoints==0.2.2
et-xmlfile==1.0.1
fastcache==1.0.2
filelock==2.0.6
Flask==0.12.2
Flask-Cors==3.0.2
future==0.16.0
gevent==1.2.1
gitdb2==2.0.2
github3.py==1.0.0a4
GitPython==2.1.5
graphviz==0.8.1
greenlet==0.4.12
h5py==2.7.0
HeapDict==1.0.0
-e git+https://github.com/hyperopt/hyperopt-sklearn.git@4b28c67b91c67ecea32bc27d64c15b2635991336#egg=hpsklearn
html5lib==0.999999999
hyperopt==0.1
idna==2.5
imagesize==0.7.1
incremental==16.10.1
ipykernel==4.6.1
ipython==5.3.0
ipython-genutils==0.2.0
ipywidgets==6.0.0
isort==4.2.5
itsdangerous==0.24
jdcal==1.3
jedi==0.10.2
Jinja2==2.9.6
jsonschema==2.6.0
jupyter==1.0.0
jupyter-client==5.0.1
jupyter-console==5.1.0
jupyter-core==4.3.0
lazy-object-proxy==1.2.2
llvmlite==0.18.0
locket==0.2.0
lxml==3.7.3
MarkupSafe==0.23
matplotlib==2.0.2
mistune==0.7.4
mpmath==0.19
msgpack-python==0.4.8
multipledispatch==0.4.9
navigator-updater==0.1.0
nb-anacondacloud==1.2.0
nb-conda==2.0.0
nb-conda-kernels==2.0.0
nbconvert==5.1.1
nbformat==4.3.0
nbpresent==3.0.2
networkx==1.11
nltk==3.2.3
nose==1.3.7
notebook==5.0.0
numba==0.33.0
numexpr==2.6.2
numpy==1.12.1
numpydoc==0.6.0
oauthlib==2.0.1
odo==0.5.0
olefile==0.44
openpyxl==2.4.7
packaging==16.8
pandas==0.20.1
pandocfilters==1.4.1
parsel==1.1.0
partd==0.3.8
pathlib2==2.2.1
patsy==0.4.1
pep8==1.7.0
pexpect==4.2.1
pickleshare==0.7.4
Pillow==4.1.1
pkginfo==1.3.2
ply==3.10
prompt-toolkit==1.0.14
psutil==5.2.2
ptyprocess==0.5.1
py==1.4.33
pyasn1==0.1.9
pyasn1-modules==0.0.8
pycosat==0.6.2
pycparser==2.17
pycrypto==2.6.1
pycurl==7.43.0
PyDispatcher==2.0.5
pyflakes==1.5.0
Pygments==2.2.0
pylint==1.6.4
pymongo==3.6.0
pyodbc==4.0.16
pyOpenSSL==17.0.0
pyparsing==2.1.4
pytest==3.0.7
python-dateutil==2.6.0
pytz==2017.2
PyWavelets==0.5.2
PyYAML==3.12
pyzmq==16.0.2
QtAwesome==0.4.4
qtconsole==4.3.0
QtPy==1.2.1
queuelib==1.4.2
redis==2.10.5
requests==2.14.2
requests-oauthlib==0.7.0
rope-py3k==0.9.4.post1
scikit-image==0.13.0
scikit-learn==0.18.1
scipy==0.19.0
Scrapy==1.3.0
seaborn==0.7.1
service-identity==16.0.0
simplegeneric==0.8.1
simplejson==3.11.1
singledispatch==3.4.0.3
six==1.10.0
smmap2==2.0.3
snowballstemmer==1.2.1
sockjs-tornado==1.0.3
sortedcollections==0.5.3
sortedcontainers==1.5.7
Sphinx==1.5.6
spyder==3.1.4
SQLAlchemy==1.1.9
statsmodels==0.8.0
sympy==1.0
tables==3.3.0
tabulate==0.7.7
tblib==1.3.2
terminado==0.6
testpath==0.3
toolz==0.8.2
tornado==4.5.1
traitlets==4.3.2
tweepy==3.5.0
Twisted==16.6.0
unicodecsv==0.14.1
uritemplate==3.0.0
uritemplate.py==3.0.2
w3lib==1.16.0
wcwidth==0.1.7
webencodings==0.5
Werkzeug==0.12.2
widgetsnbextension==2.0.0
wrapt==1.10.10
xgboost==0.6
xlrd==1.0.0
XlsxWriter==0.9.6
xlwings==0.10.4
xlwt==1.2.0
zict==0.1.2
zope.interface==4.3.3
```
