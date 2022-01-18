# Running RAMP-DA on Linux servers

## Log in 

Log in to main server (don't run anything on this computer!)

```
ssh [username]@see-gw-01.leeds.ac.uk
```

Log in to linux machines. There are four of them (`foe-linux-01` - `foe-linux-04`) so you can log into each one individually, or just use whichever one is the quietest with:

```
ssh foe-linux
```

## Install local anaconda

Instructions are [here](https://docs.anaconda.com/anaconda/install/linux/).

I started by making a new directory for the install but this is optional
```
mkdir anaconda_install
cd anaconda_install
```

Download the Linux installer (I found the link on their [website](https://www.anaconda.com/products/individual#linux))
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
```

Run the installer (first changing the permissions to allow you to execute the file

```
chmod +x Anaconda3-2021.11-Linux-x86_64.sh 
./Anaconda3-2021.11-Linux-x86_64.sh 
```

The installer will ask some questions and the install. When it asks about initialising with 'conda init' choose "yes".

Finally 'activate' (not the right word) the installation with:
```
source ~/.bashrc.
```
(only needs to be done once as the `.bashrc` file is read each time you log in).

Could check that `conda` points to the correct location (it should be somewhere in `~/anaconda3`:

```
which conda
~/anaconda3/bin/conda
```

## Create the environment

I couldn't use the environment file, so did it manually a few packages at a time (calling ithe environment `ramp-ua-min` here)

Create the environment:

```conda create --name ramp-ua-min -c conda-forge python=3.7 pyopencl=2020.2.2 numpy pandas=1.0.3 geopandas pocl matplotlib```

Then activate and install the remaining packages. These may be able to be installed all at once, but I install the main ones first (which I was sure we would need) then added others as needed by running the model and seeing which libraries it asked for).

Note: if you want to run all the conda commands at once, put `-y` after `conda install` to automatically answer yes to all the questions, and then a semicolon after each line below. Then you can do one big copy-and-paste run them all one after the other

```conda activate ramp-ua-min```

```conda install -c conda-forge tqdm click swifter tzlocal imageio pyyaml ```

```conda install -c conda-forge scipy click descartes```

```conda install -c conda-forge rpy2```

```pip install glfw``` (not sure why need to use `pip` for that one, but if you use conda it can't find the library at runtime)

```pip install imgui``` (that one is only available with `pip`)

```pip install pyopengl```

```pip install convertbng```

```conda install -c conda-forge plotly pygam```

```pip install pyabc```


## Initialise the model 

First time set up of library paths

```
python setup.py develop
```

Download the Devon data and create initialise the population

```
python microsim/main.py -init
```

If anything goes wrong during the devon data download, delete these files before trying again:

 - `devon_data.tar.gz`
 - `devon_data/`

If initialisation fails (after the data download) try deleting (if they exist):

 - `devon_data/caches/*.pkl`
 - `microsim/opencl/snapshots/cache.npz`

Run the OpenCL version of the model to generate an opencl snapshot of the population

```
python microsim/main.py -ocl
```

Done!!!

A couple of minor things to note:

 - The only other thing to do to make `molly_test.py` work is make sure that `usegpu` is set to False. Otherwise it tries to find a GPU but can't.
 - If you use the multi-process pyabc sampler things should run really quickly
 - If you want to make sure the script keeps running even if you log out (so you don't need to leave your laptop on to maintain the connection to the server) you can use `nohup` like this: `nohup python molly_test.py &`. 
   - The output will be written to a file called `nohup.out` by default. Then doing something like `tail -f nohup.out` will show you the contents of the file and update automatically. If you want the output to go somewhere else, e.g. if you were running slightly different versions of the script at once, you can do something like `nohup python molly_test.py > outputfile.txt &`
   - Sometimes the print statements don't make their way into the output file for some time. To get round this, include `flush=True` in any print statement, and it will force python to print the output. I added it to line 779 in `opencl_runner.py`. e.g.: `print("OpenclRunner ran model {} in {}".format(model_number, datetime.datetime.now() - start_time), flush=True)`
   - When using `nohup`, to stop the model running you need to find it's ID using `ps` and then use the `kill` command like so:
```
(ramp-ua-min) [foe-linux-03.leeds.ac.uk:geonsm:128]$ ps
   PID TTY          TIME CMD
105510 pts/183  00:07:31 python
120519 pts/183  00:00:00 ps
176744 pts/183  00:00:02 bash
(ramp-ua-min) [foe-linux-03.leeds.ac.uk:geonsm:129]$ kill 105510
```

