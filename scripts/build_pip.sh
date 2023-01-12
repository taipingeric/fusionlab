#!/bin/bash
# ref: https://datainpoint.substack.com/p/c0a

python ../setup.py sdist bdist_wheel
python -m twine upload ../dist/*
