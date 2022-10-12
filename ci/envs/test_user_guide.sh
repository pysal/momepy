#!/bin/bash -e

echo "Testing User Guide"

python -m ipykernel install --user --name geo_dev

mkdir docs/user_guide/output

jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1000 docs/user_guide/getting_started.ipynb --output-dir docs/user_guide/output
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1000 docs/user_guide/elements/*.ipynb --output-dir docs/user_guide/output
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1000 docs/user_guide/simple/*.ipynb --output-dir docs/user_guide/output
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1000 docs/user_guide/combined/*.ipynb --output-dir docs/user_guide/output
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1000 docs/user_guide/weights/*.ipynb --output-dir docs/user_guide/output
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1000 docs/user_guide/graph/convert.ipynb --output-dir docs/user_guide/output
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1000 docs/user_guide/graph/network.ipynb --output-dir docs/user_guide/output
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1000 docs/user_guide/graph/coins.ipynb --output-dir docs/user_guide/output
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1000 docs/user_guide/preprocessing/*.ipynb --output-dir docs/user_guide/output

rm -rf docs/user_guide/output
