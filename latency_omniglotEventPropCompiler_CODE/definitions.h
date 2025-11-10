#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <random>
#include <string>
#include <stdexcept>

// Standard C includes
#include <cstdint>
// CUDA includes
#include <curand_kernel.h>
#include <cuda_fp16.h>

// ------------------------------------------------------------------------
// Helper macro for error-checking CUDA calls
#define CHECK_RUNTIME_ERRORS(call) {\
    cudaError_t error = call;\
    if (error != cudaSuccess) {\
        throw std::runtime_error(__FILE__": " + std::to_string(__LINE__) + ": cuda error " + std::to_string(error) + ": " + cudaGetErrorString(error));\
    }\
}

#include <cassert>

struct XORWowStateInternal
 {
    unsigned int d;
    unsigned int v[5];
}
;

template<typename RNG>
__device__ inline float exponentialDistFloat(RNG *rng) {
    while (true) {
        const float u = curand_uniform(rng);
        if (u != 0.0f) {
            return -logf(u);
        }
    }
}

template<typename RNG>
__device__ inline double exponentialDistDouble(RNG *rng) {
    while (true) {
        const double u = curand_uniform_double(rng);
        if (u != 0.0) {
            return -log(u);
        }
    }
}

template<typename RNG>
__device__ inline float gammaDistFloatInternal(RNG *rng, float c, float d)
 {
    float x, v, u;
    while (true) {
        do {
            x = curand_normal(rng);
            v = 1.0f + c*x;
        }
        while (v <= 0.0f);
        
        v = v*v*v;
        do {
            u = curand_uniform(rng);
        }
        while (u == 1.0f);
        
        if (u < 1.0f - 0.0331f*x*x*x*x) break;
        if (logf(u) < 0.5f*x*x + d*(1.0f - v + logf(v))) break;
    }
    
    return d*v;
}

template<typename RNG>
__device__ inline float gammaDistFloat(RNG *rng, float a)
 {
    if (a > 1)
     {
        const float u = curand_uniform (rng);
        const float d = (1.0f + a) - 1.0f / 3.0f;
        const float c = (1.0f / 3.0f) / sqrtf(d);
        return gammaDistFloatInternal (rng, c, d) * powf(u, 1.0f / a);
    }
    else
     {
        const float d = a - 1.0f / 3.0f;
        const float c = (1.0f / 3.0f) / sqrtf(d);
        return gammaDistFloatInternal(rng, c, d);
    }
}

template<typename RNG>
__device__ inline float gammaDistDoubleInternal(RNG *rng, double c, double d)
 {
    double x, v, u;
    while (true) {
        do {
            x = curand_normal_double(rng);
            v = 1.0 + c*x;
        }
        while (v <= 0.0);
        
        v = v*v*v;
        do {
            u = curand_uniform_double(rng);
        }
        while (u == 1.0);
        
        if (u < 1.0 - 0.0331*x*x*x*x) break;
        if (log(u) < 0.5*x*x + d*(1.0 - v + log(v))) break;
    }
    
    return d*v;
}

template<typename RNG>
__device__ inline float gammaDistDouble(RNG *rng, double a)
 {
    if (a > 1.0)
     {
        const double u = curand_uniform (rng);
        const double d = (1.0 + a) - 1.0 / 3.0;
        const double c = (1.0 / 3.0) / sqrt(d);
        return gammaDistDoubleInternal (rng, c, d) * pow(u, 1.0 / a);
    }
    else
     {
        const float d = a - 1.0 / 3.0;
        const float c = (1.0 / 3.0) / sqrt(d);
        return gammaDistDoubleInternal(rng, c, d);
    }
}

template<typename RNG>
__device__ inline unsigned int binomialDistFloatInternal(RNG *rng, unsigned int n, float p)
 {
    const float q = 1.0f - p;
    const float qn = expf(n * logf(q));
    const float np = n * p;
    const unsigned int bound = min(n, (unsigned int)(np + (10.0f * sqrtf((np * q) + 1.0f))));
    unsigned int x = 0;
    float px = qn;
    float u = curand_uniform(rng);
    while(u > px)
     {
        x++;
        if(x > bound) {
            x = 0;
            px = qn;
            u = curand_uniform(rng);
        }
        else {
            u -= px;
            px = ((n - x + 1) * p * px) / (x * q);
        }
    }
    return x;
}

template<typename RNG>
__device__ inline unsigned int binomialDistFloat(RNG *rng, unsigned int n, float p)
 {
    if(p <= 0.5f) {
        return binomialDistFloatInternal(rng, n, p);
    }
    else {
        return (n - binomialDistFloatInternal(rng, n, 1.0f - p));
    }
}
template<typename RNG>
__device__ inline unsigned int binomialDistDoubleInternal(RNG *rng, unsigned int n, double p)
 {
    const double q = 1.0 - p;
    const double qn = exp(n * log(q));
    const double np = n * p;
    const unsigned int bound = min(n, (unsigned int)(np + (10.0 * sqrt((np * q) + 1.0))));
    unsigned int x = 0;
    double px = qn;
    double u = curand_uniform_double(rng);
    while(u > px)
     {
        x++;
        if(x > bound) {
            x = 0;
            px = qn;
            u = curand_uniform_double(rng);
        }
        else {
            u -= px;
            px = ((n - x + 1) * p * px) / (x * q);
        }
    }
    return x;
}

template<typename RNG>
__device__ inline unsigned int binomialDistDouble(RNG *rng, unsigned int n, double p)
 {
    if(p <= 0.5) {
        return binomialDistDoubleInternal(rng, n, p);
    }
    else {
        return (n - binomialDistDoubleInternal(rng, n, 1.0 - p));
    }
}
#define SCALAR_MIN 1.175494351e-38f
#define SCALAR_MAX 3.402823466e+38f
#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f
extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
extern __device__ curandStatePhilox4_32_10_t d_rng;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double initTime;
EXPORT_VAR double initSparseTime;
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
EXPORT_VAR double customUpdateBatchSoftmax1Time;
EXPORT_VAR double customUpdateBatchSoftmax1TransposeTime;
EXPORT_VAR double customUpdateBatchSoftmax1RemapTime;
EXPORT_VAR double customUpdateBatchSoftmax2Time;
EXPORT_VAR double customUpdateBatchSoftmax2TransposeTime;
EXPORT_VAR double customUpdateBatchSoftmax2RemapTime;
EXPORT_VAR double customUpdateBatchSoftmax3Time;
EXPORT_VAR double customUpdateBatchSoftmax3TransposeTime;
EXPORT_VAR double customUpdateBatchSoftmax3RemapTime;
EXPORT_VAR double customUpdateGradientBatchReduceTime;
EXPORT_VAR double customUpdateGradientBatchReduceTransposeTime;
EXPORT_VAR double customUpdateGradientBatchReduceRemapTime;
EXPORT_VAR double customUpdateGradientLearnTime;
EXPORT_VAR double customUpdateGradientLearnTransposeTime;
EXPORT_VAR double customUpdateGradientLearnRemapTime;
EXPORT_VAR double customUpdateResetTime;
EXPORT_VAR double customUpdateResetTransposeTime;
EXPORT_VAR double customUpdateResetRemapTime;
EXPORT_VAR double customUpdateZeroGradientTime;
EXPORT_VAR double customUpdateZeroGradientTransposeTime;
EXPORT_VAR double customUpdateZeroGradientRemapTime;
EXPORT_VAR double customUpdateZeroOutPostTime;
EXPORT_VAR double customUpdateZeroOutPostTransposeTime;
EXPORT_VAR double customUpdateZeroOutPostRemapTime;
// Runner functions
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC void stepTime(unsigned long long timestep, unsigned long long numRecordingTimesteps);

// Functions generated by backend
EXPORT_FUNC void updateNeurons(float t); 
EXPORT_FUNC void updateSynapses(float t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
EXPORT_FUNC void initializeHost();
EXPORT_FUNC void updateBatchSoftmax1(unsigned long long timestep);
EXPORT_FUNC void updateBatchSoftmax2(unsigned long long timestep);
EXPORT_FUNC void updateBatchSoftmax3(unsigned long long timestep);
EXPORT_FUNC void updateGradientBatchReduce(unsigned long long timestep);
EXPORT_FUNC void updateGradientLearn(unsigned long long timestep);
EXPORT_FUNC void updateReset(unsigned long long timestep);
EXPORT_FUNC void updateZeroGradient(unsigned long long timestep);
EXPORT_FUNC void updateZeroOutPost(unsigned long long timestep);

// Merged group upload functions
EXPORT_FUNC void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, float* LambdaI, float* LambdaV, float* Softmax, float* V, float* VAvg, uint8_t* YTrue, uint8_t* YTrueBack, float* outPostInSyn0);
EXPORT_FUNC void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, uint8_t* BackSpike, float* LambdaI, float* LambdaV, int32_t* RingReadEndOffset, int32_t* RingReadOffset, int32_t* RingWriteOffset, int32_t* RingWriteStartOffset, float* V, float* outPostInSyn0, float* outPreOutSyn0, uint32_t* spkCntEventSynSpikeEvent0, uint32_t* spkCntSynSpike0, uint32_t* spkEventSynSpikeEvent0, uint32_t* spkSynSpike0);
EXPORT_FUNC void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, uint8_t* BackSpike, uint32_t* EndSpike, int32_t* RingReadEndOffset, int32_t* RingReadOffset, int32_t* RingWriteOffset, int32_t* RingWriteStartOffset, uint32_t* StartSpike, uint32_t* spkCntEventSynSpikeEvent0, uint32_t* spkCntSynSpike0, uint32_t* spkEventSynSpikeEvent0, uint32_t* spkSynSpike0);
EXPORT_FUNC void pushMergedSynapseInitGroup0ToDevice(unsigned int idx, float* Gradient, float* g, float meang, uint32_t numSrcNeurons, uint32_t numTrgNeurons, uint32_t rowStride, float sdg);
EXPORT_FUNC void pushMergedCustomUpdateInitGroup0ToDevice(unsigned int idx, float* SumExpVal);
EXPORT_FUNC void pushMergedCustomUpdateInitGroup1ToDevice(unsigned int idx, float* MaxVal);
EXPORT_FUNC void pushMergedCustomWUUpdateInitGroup0ToDevice(unsigned int idx, float* M, float* V, uint32_t numSrcNeurons, uint32_t numTrgNeurons, uint32_t rowStride);
EXPORT_FUNC void pushMergedCustomWUUpdateInitGroup1ToDevice(unsigned int idx, float* ReducedGradient, uint32_t numSrcNeurons, uint32_t numTrgNeurons, uint32_t rowStride);
EXPORT_FUNC void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, float* LambdaI, float* LambdaV, float* Softmax, float* V, float* VAvg, uint8_t* YTrueBack, float* outPostInSyn0, uint32_t Trial);
EXPORT_FUNC void pushMergedNeuronUpdate0TrialToDevice(unsigned int idx, uint32_t value);
EXPORT_FUNC void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, uint8_t* BackSpike, float* LambdaI, float* LambdaV, float* RingIMinusV, int32_t* RingReadEndOffset, int32_t* RingReadOffset, float* RingSpikeTime, int32_t* RingWriteOffset, float* V, float* outPostInSyn0, float* outPreOutSyn0, uint32_t* spkCntEventSynSpikeEvent0, uint32_t* spkCntSynSpike0, uint32_t* spkEventSynSpikeEvent0, uint32_t* spkSynSpike0);
EXPORT_FUNC void pushMergedNeuronUpdate1RingIMinusVToDevice(unsigned int idx, float* value);
EXPORT_FUNC void pushMergedNeuronUpdate1RingSpikeTimeToDevice(unsigned int idx, float* value);
EXPORT_FUNC void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, uint8_t* BackSpike, uint32_t* EndSpike, int32_t* RingReadEndOffset, int32_t* RingReadOffset, float* RingSpikeTime, int32_t* RingWriteOffset, float* SpikeTimes, uint32_t* StartSpike, uint32_t* spkCntEventSynSpikeEvent0, uint32_t* spkCntSynSpike0, uint32_t* spkEventSynSpikeEvent0, uint32_t* spkSynSpike0);
EXPORT_FUNC void pushMergedNeuronUpdate2RingSpikeTimeToDevice(unsigned int idx, float* value);
EXPORT_FUNC void pushMergedNeuronUpdate2SpikeTimesToDevice(unsigned int idx, float* value);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* Gradient, float* LambdaI_post, float* LambdaV_post, float* g, float* outPost, float* outPre, uint32_t* srcSpk, uint32_t* srcSpkCnt, uint32_t* srcSpkCntEvent, uint32_t* srcSpkEvent);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, float* Gradient, float* LambdaI_post, float* g, float* outPost, uint32_t* srcSpk, uint32_t* srcSpkCnt, uint32_t* srcSpkCntEvent, uint32_t* srcSpkEvent);
EXPORT_FUNC void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, uint32_t* spkCntEventSynSpikeEvent0, uint32_t* spkCntSynSpike0);
EXPORT_FUNC void pushMergedCustomUpdateGroup0ToDevice(unsigned int idx, float* OutPost, uint32_t numNeurons);
EXPORT_FUNC void pushMergedCustomUpdateGroup1ToDevice(unsigned int idx, float* MaxVal, float* SoftmaxVal, float* SumExpVal, float* Val);
EXPORT_FUNC void pushMergedCustomUpdateGroup2ToDevice(unsigned int idx, float* MaxVal, float* SumExpVal, float* Val);
EXPORT_FUNC void pushMergedCustomUpdateGroup3ToDevice(unsigned int idx, float* MaxVal, float* Val);
EXPORT_FUNC void pushMergedCustomUpdateGroup4ToDevice(unsigned int idx, uint8_t* YTrue, uint8_t* YTrueBack);
EXPORT_FUNC void pushMergedCustomUpdateGroup5ToDevice(unsigned int idx, float* LambdaI, float* LambdaV, float* V, float* VAvg);
EXPORT_FUNC void pushMergedCustomUpdateGroup6ToDevice(unsigned int idx, float* LambdaI, float* LambdaV, int32_t* RingReadEndOffset, int32_t* RingReadOffset, int32_t* RingWriteOffset, int32_t* RingWriteStartOffset, float* V);
EXPORT_FUNC void pushMergedCustomUpdateGroup7ToDevice(unsigned int idx, int32_t* RingReadEndOffset, int32_t* RingReadOffset, int32_t* RingWriteOffset, int32_t* RingWriteStartOffset, uint32_t* StartSpike);
EXPORT_FUNC void pushMergedCustomUpdateWUGroup0ToDevice(unsigned int idx, float* Gradient, uint32_t numSrcNeurons, uint32_t rowStride);
EXPORT_FUNC void pushMergedCustomUpdateWUGroup1ToDevice(unsigned int idx, float* Gradient, float* M, float* V, float* Variable, float Alpha, float MomentScale1, float MomentScale2, uint32_t numSrcNeurons, uint32_t rowStride);
EXPORT_FUNC void pushMergedCustomUpdateWU1AlphaToDevice(unsigned int idx, float value);
EXPORT_FUNC void pushMergedCustomUpdateWU1MomentScale1ToDevice(unsigned int idx, float value);
EXPORT_FUNC void pushMergedCustomUpdateWU1MomentScale2ToDevice(unsigned int idx, float value);
EXPORT_FUNC void pushMergedCustomUpdateWUGroup2ToDevice(unsigned int idx, float* Gradient, float* ReducedGradient, uint32_t numSrcNeurons, uint32_t rowStride);
}  // extern "C"
