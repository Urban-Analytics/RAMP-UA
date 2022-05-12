# Ceate the environment
```conda create --name ramp-ua-min -c conda-forge python=3.7 pyopencl=2020.2.2 numpy pandas=1.0.3 geopandas matplotlib```

Then activate and install the remaining packages. These may be able to be installed all at once, but I install the main ones first (which I was sure we would need) then added others as needed by running the model and seeing which libraries it asked for).


pip install pyabc
conda install jupyter notebook

# To install pocl
Download the wheel from:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl
For me it was:
pyopencl-2022.1-cp37-cp37m-win_amd64.whl
then pip install C:/Users/gy17m2a/Downloads/pyopencl-2022.1-cp37-cp37m-win_amd64.whl

pip import glfw
pip install imgui
pip isntall pyopengl


ramp-ua-min
conda create --name ramp-ua-min -c conda-forge python=3.7 pyopencl=2020.2.2 numpy pandas=1.0.3 geopandas matplotlib

rpy2 wheel? was this necessary?
os.environ['R_HOME'] = 'C:/Users/gy17m2a/AppData/Local/Programs/R/R-4.2.0' #path to your R installation
os.environ['R_USER'] = 'C:/ProgramData/Anaconda3/envs/analyse_results/Lib/site-packages/rpy2' #path depends on where you installed Python. Mine is the Anaconda distribution
