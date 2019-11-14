. ./.github/workflows/set_os_env.sh

echo "activate test-environment =================="
source $MINICONDA_SUB_PATH/activate test
$MINICONDA_SUB_PATH/conda list

$MINICONDA_PYTEST --numprocesses=auto tests/
