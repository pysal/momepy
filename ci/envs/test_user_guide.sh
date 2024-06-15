#!/bin/bash -e

echo "Testing User Guide"

python -m ipykernel install --user --name geo_dev

mkdir docs/user_guide/output

jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 docs/user_guide/getting_started.ipynb --output-dir docs/user_guide/output
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 docs/user_guide/*/*.ipynb --output-dir docs/user_guide/output
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 docs/examples/*.ipynb --output-dir docs/user_guide/output

rm -rf docs/user_guide/output
