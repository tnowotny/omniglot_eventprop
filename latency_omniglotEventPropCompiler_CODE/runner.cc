#include "definitions.h"

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
__device__ curandStatePhilox4_32_10_t d_rng;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double initTime = 0.0;
double initSparseTime = 0.0;
double neuronUpdateTime = 0.0;
double presynapticUpdateTime = 0.0;
double postsynapticUpdateTime = 0.0;
double synapseDynamicsTime = 0.0;
double customUpdateBatchSoftmax1Time = 0.0;
double customUpdateBatchSoftmax1TransposeTime = 0.0;
double customUpdateBatchSoftmax1RemapTime = 0.0;
double customUpdateBatchSoftmax2Time = 0.0;
double customUpdateBatchSoftmax2TransposeTime = 0.0;
double customUpdateBatchSoftmax2RemapTime = 0.0;
double customUpdateBatchSoftmax3Time = 0.0;
double customUpdateBatchSoftmax3TransposeTime = 0.0;
double customUpdateBatchSoftmax3RemapTime = 0.0;
double customUpdateGradientBatchReduceTime = 0.0;
double customUpdateGradientBatchReduceTransposeTime = 0.0;
double customUpdateGradientBatchReduceRemapTime = 0.0;
double customUpdateGradientLearnTime = 0.0;
double customUpdateGradientLearnTransposeTime = 0.0;
double customUpdateGradientLearnRemapTime = 0.0;
double customUpdateResetTime = 0.0;
double customUpdateResetTransposeTime = 0.0;
double customUpdateResetRemapTime = 0.0;
double customUpdateZeroGradientTime = 0.0;
double customUpdateZeroGradientTransposeTime = 0.0;
double customUpdateZeroGradientRemapTime = 0.0;
double customUpdateZeroOutPostTime = 0.0;
double customUpdateZeroOutPostTransposeTime = 0.0;
double customUpdateZeroOutPostRemapTime = 0.0;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
}  // extern "C"
void allocateMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
}

void stepTime(unsigned long long timestep, unsigned long long numRecordingTimesteps) {
    const float t = timestep * 1.000000000e+00f;
    updateSynapses(t);
    updateNeurons(t); 
}

