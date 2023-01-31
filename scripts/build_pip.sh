#!/bin/bash
# ref: https://datainpoint.substack.com/p/c0a

python ../setup.py sdist bdist_wheel
python -m twine upload ../dist/*

# build
python setup.py sdist bdist_wheel
# pypi
twine upload dist/*
# testpypi
twine upload --repository testpypi dist/*

# testpypi install
!pip install --upgrade fusionlab==?
!pip install -i https://test.pypi.org/simple/ --upgrade fusionlab==?
