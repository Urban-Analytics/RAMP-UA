


echo "Setting up your colab environment to run RAMP-UA"
echo "********************************************************************************************************"

install_ramp () {
    # install miniconda
    echo "Installing conda"
    wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
    chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
    bash ./Miniconda3-4.5.4-Linux-x86_64.sh -b -f -p /usr/local

    conda env update -p /usr/local -f RAMP-UA/examples/colab/ramp-colab.yml

    cd RAMP-UA

    python setup.py install

    echo "RAMP-UA conda environment created!"

}

install_ramp