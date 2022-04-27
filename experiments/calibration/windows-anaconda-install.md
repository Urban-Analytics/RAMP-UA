# Ceate the environment
```conda create --name ramp-ua-min -c conda-forge python=3.7 pyopencl=2020.2.2 numpy pandas=1.0.3 geopandas matplotlib```

Then activate and install the remaining packages. These may be able to be installed all at once, but I install the main ones first (which I was sure we would need) then added others as needed by running the model and seeing which libraries it asked for).

pip install pyopencl[pocl]
