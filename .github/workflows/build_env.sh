. ./.github/workflows/set_os_env.sh

echo "exporting a new path ======================="
export PATH="$MINICONDA_PATH:$MINICONDA_SUB_PATH:$PATH"

if [[ $RUNNER_OS ==  "macOS" ]]; then
    . ./.github/workflows/wget_install_miniconda.sh
fi

echo "init conda ================================="
$MINICONDA_SUB_PATH/conda init bash
echo "~/$BASHRC =================================="
. ~/$BASHRC
echo "hash -r ===================================="
hash -r
echo "============================================"
echo "CONDA_PYTHON = $CONDA_PYTHON ==============="
echo "============================================"
echo "checking python version ===================="
$MINICONDA_PATH/python --version
echo "conda config --yes ========================="
$MINICONDA_SUB_PATH/conda config --set always_yes yes --set changeps1 no;
echo "conda update ==============================="
$MINICONDA_SUB_PATH/conda update -q conda;
echo "conda info -a =============================="
$MINICONDA_SUB_PATH/conda info -a
echo "create test-environment ===================="
$MINICONDA_SUB_PATH/conda env create -f ./ci/travis/37-mc.yaml
