#!/bin/bash
cd "$(dirname "$0")"/..
jupyter nbconvert --no-input --output-dir build --to html notebook/confinement.ipynb 

