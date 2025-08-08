#!/bin/sh
cd cpp
python -m mkdocs build
cd ..
cd py
python -m mkdocs build
cd ..
rm -rf docs/*
cp -r cpp/site_build/* docs/
cp -r py/site_build docs/py/
git add docs/