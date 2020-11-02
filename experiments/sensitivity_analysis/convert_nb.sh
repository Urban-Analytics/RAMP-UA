jupyter nbconvert --to script sensitivity_analysis.ipynb
echo "Script converted, running ... "
nohup python sensitivity_analysis.py &
