# uncertainty_quant_peract

1. Install CoppeliaSim
2. Create and activate a virtual environment, install PyRep, RLBench, YARR accordingly in the submodule
3. Add path Go to "$path/to/uncertainty_quant_peract"
   cd "$path/to/uncertainty_quant_peract/peract_reliability"
   export PERACT_ROOT=$(pwd)
5. Download ckpts for Perceiver-Actor, data for confidence calibration, data for task evaluation
6. Run temperature scaling calibration on calibration data
7. Run task evaluation on evaluation data
