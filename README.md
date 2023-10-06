# uncertainty_quant_peract

1. Create and activate a virtual environment
   ```
   # use virtualenv to create and activate the environment
   virtualenv -p $(which python3.8) --system-site-packages peract_env  
   source peract_env/bin/activate
   pip install --upgrade pip

   # update all submodules
   git submodule update --init --recursive
   ```
2. After cloning all submodules, install PyRep, RLBench, YARR accordingly in the submodule according to instructions in [Perceiver-Actor](https://github.com/peract/peract).
  
   Note you don't need to ```git clone``` the packages in the installation instructions again, they have been included in the submodules and are heavily modified to add support for the uncertainty aware action selection strategy method.

4. 
2. Add path Go to "$path/to/uncertainty_quant_peract"
   ```
   cd $path/to/uncertainty_quant_peract/peract_reliability
   export PERACT_ROOT=$(pwd)
   ```
3. Download ckpts for Perceiver-Actor, data for confidence calibration, data for task evaluation
4. Run temperature scaling calibration on calibration data
5. Run task evaluation on evaluation data
