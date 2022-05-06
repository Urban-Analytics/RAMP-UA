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
pip install glfw imgui pyopengl
