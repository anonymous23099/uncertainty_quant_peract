# uncertainty_quant_peract

1. Create and activate a virtual environment, install PyRep, RLBench, YARR accordingly in the submodule
   ```
   # use virtualenv to create and activate the environment
   virtualenv -p $(which python3.8) --system-site-packages peract_env  
   source peract_env/bin/activate
   pip install --upgrade pip

   # update all submodules
   git submodule update --init --recursive

   # after cloning all submodules, install all packages
   Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:
   ```
      PyRep requires version **4.1** of CoppeliaSim. Download: 
   - [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
   - [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
   - [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)
  
     ```
     export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
     export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
     ```
2. Add path Go to "$path/to/uncertainty_quant_peract"
   ```
   cd $path/to/uncertainty_quant_peract/peract_reliability
   export PERACT_ROOT=$(pwd)
   ```
3. Download ckpts for Perceiver-Actor, data for confidence calibration, data for task evaluation
4. Run temperature scaling calibration on calibration data
5. Run task evaluation on evaluation data
