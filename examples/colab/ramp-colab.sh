


echo "Setting up your colab environment to run RAMP-UA"
echo "********************************************************************************************************"

install_ramp () {

    # apt get various things
    apt-get install -y git-lfs 
    apt install -y libspatialindex-dev

    # install git lfs
    git lfs install

    # clone in repository from a branch named as argument
    git clone --single-branch --branch $RESPONSE https://github.com/Urban-Analytics/RAMP-UA.git

    if [ ! -d RAMP-UA ]; then
        echo "Git clone appears to have failed."
        exit 1

    # download the specific notebook functions file
    curl -O https://raw.githubusercontent.com/Urban-Analytics/RAMP-UA/master/experiments/functions.py

    # download cache file
    curl -o default.npz https://zenodo.org/record/4153512/files/default.npz?download=1

    # install miniconda
    echo "Installing miniconda"
    export PYTHONPATH=
    wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
    chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
    bash ./Miniconda3-4.5.4-Linux-x86_64.sh -b -f -p /usr/local

    echo "python 3.6.*" > /usr/local/conda-meta/pinned

    #Installing another conda package first something first seems to fix https://github.com/rapidsai/rapidsai-csp-utils/issues/4
    conda install --channel defaults conda python=3.6 --yes
    conda update -y -c conda-forge -c defaults --all
    conda install -y --prefix /usr/local -c conda-forge -c defaults openssl six

    conda install -y --prefix /usr/local \
            -c conda-forge -c defaults \
            python=3.6 pandas=1.0.3 matplotlib=3.1.3 pyopencl=2020.2.2 \
            click pyyaml rpy2=3.3.2 numpy=1.18.5 tqdm scipy=1.5.2 \
            swifter=0.304 tzlocal r-mixdist r-tidyr=1.1.0 r-stringr \
            r-readr r-curl imageio=2.8.0 r-janitor=2.0.1 r-stringi=1.4.6 r-devtools=2.3.1 \
            r-data.table r-mixdist r-mgcv r-RecordLinkage r-tidyselect r-rvcheck r-stringdist pocl \
            geopandas=0.7.0 descartes ocl-icd-system

    pip install -r RAMP-UA/microsim/opencl/requirements.txt
    pip install convertbng

    cd RAMP-UA/

    pip install .

    python -c "sys.path.append("/usr/local/lib/python3.6/site-packages")"

    echo "RAMP-UA conda environment created!"


}

if [ -n "$1" ] ; then
    RESPONSE=$1
    install_ramp
else
    echo "Please pass the branch of RAMP-UA you wish to install in this notebook as an argument to this function."
fi