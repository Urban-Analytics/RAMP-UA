jupyter nbconvert --to script calibration.ipynb
echo "Script converted, running ... "
nohup python calibration.py &
