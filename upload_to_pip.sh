#!/usr/bin/env bash
rm ./dist/*
python3 -m build
python3 -m twine upload --repository pypi dist/* --verbose
