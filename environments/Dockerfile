FROM jupyter/minimal-notebook:latest

LABEL maintainer="Martin Fleischmann <martin@martinfleischmann.net>"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN conda install mamba \
    && conda config --add channels conda-forge \
    && conda config --set channel_priority strict \
    && mamba install --yes --quiet \
    'geopandas' \
    'momepy' \
    'seaborn' \
    'scikit-learn' \
    'contextily' \
    'legendgram' \
    'pysal' \
    'osmnx' \
    'scipy' \
    'clustergram' \
    'dask' \
    && conda clean --all --yes --force-pkgs-dirs \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete

#--- Jupyter config ---#
USER root
RUN echo "c.NotebookApp.default_url = '/lab'"\
    >> /home/$NB_USER/.jupyter/jupyter_notebook_config.py \
    && jupyter lab build -y \
    # Clean cache up
    && jupyter lab clean -y \
    && conda clean --all -f -y \
    && npm cache clean --force \
    && rm -rf $CONDA_DIR/share/jupyter/lab/staging \
    && rm -rf "/home/${NB_USER}/.node-gyp" \
    && rm -rf /home/$NB_USER/.cache/yarn \
    # Fix permissions
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}" \
    # Build mpl font cache
    && python -c "import matplotlib.pyplot;"
USER $NB_UID

#--- htop ---#

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to user to avoid accidental container runs as root
USER $NB_UID