# Ceate the environment

### Initialise the environment:

```conda create --name ramp-ua-min -c conda-forge python=3.7 pyopencl=2020.2.2 numpy pandas=1.0.3 geopandas matplotlib```

### Activate the environment

```conda activate ramp-ua-min```

### Install pocl
Download the wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl  
Selecting correct version for python version and whether computer is 32 or 64 bit. E.g. for Python 3.7 and 64 bit system then choose 
```pyopencl-2022.1-cp37-cp37m-win_amd64.whl```  ("Settings -> System -> About" to check whether 32 or 64 bit)

Then ```pip install C:/Users/gy17m2a/Downloads/pyopencl-2022.1-cp37-cp37m-win_amd64.whl```

### Install further packages
```pip install pyabc```  
```conda install jupyter notebook```  
```pip install glfw```  
```pip install imgui```  
```pip install pyopengl```  

### Install rpy2
rpy2 wheel? was this necessary?  
In ```microsim/main.py``` add the following to the script:  
```os.environ['R_HOME'] = 'C:/Users/gy17m2a/AppData/Local/Programs/R/R-4.2.0'``` #path to your R installation  
```os.environ['R_USER'] = 'C:/ProgramData/Anaconda3/envs/analyse_results/Lib/site-packages/rpy2'``` #path depends on where you installed Python. Mine is the Anaconda distribution    
