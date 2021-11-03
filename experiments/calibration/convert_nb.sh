jupyter nbconvert --to script $1
echo "Script $1 converted"
# These lines run the script (I've turned this off for now)
# (although it wont work because '.ipynb' needs to be stripped from $1)
#echo "Running $1.py "
#nohup python $1.py &
