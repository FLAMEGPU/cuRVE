
#include <stdio.h>
#include "curve.cuh"
#include "assert.h"

#define CURVE_MAX_VARIABLES 			32 							//!< Default maximum number of cuRVE variables (must be a power of 2)
#define VARIABLE_DISABLED 				0
#define VARIABLE_ENABLED 				1
#define NAMESPACE_NONE 					0

#ifdef DEBUG
#define CUDA_SAFE_CALL(x)                                                                               		\
{                                                                                                         		\
	cudaError_t error = (x);                                                                                	\
	if (error != cudaSuccess && error != cudaErrorNotReady)                                                 	\
	{                                                                                                       	\
		printf("%s.%s.%d: 0x%x (%s)\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error));  	\
		cudaGetLastError();                                                                                   	\
		exit(1);                                                                                              	\
	}                                                                                                       	\
}
#else
#define CUDA_SAFE_CALL(x) (x)
#endif




inline void cudaCheckError(cudaError_t error, char* file, char* function, int line){
	if (error != cudaSuccess && error != cudaErrorNotReady)
	{
		printf("%s.%s.%d: 0x%x (%s)\n", file, function, line, error, cudaGetErrorString(error));
		cudaGetLastError();
		exit(1);
	}
}


unsigned int h_default_vector_width;

unsigned int h_namespace;

CurveVariableHash h_hashes[CURVE_MAX_VARIABLES];
float* h_variables[CURVE_MAX_VARIABLES];
float* h_d_variables[CURVE_MAX_VARIABLES];
int	h_states[CURVE_MAX_VARIABLES];

__constant__ CurveNamespaceHash d_namespace;
__constant__ CurveVariableHash d_hashes[CURVE_MAX_VARIABLES];
__device__ float* d_variables[CURVE_MAX_VARIABLES];
__constant__ int* d_states[CURVE_MAX_VARIABLES];

__device__ curveDeviceError d_curve_error;
curveHostError h_curve_error;


/*private functions*/

__device__ __inline__ CurveVariable getVariable(const CurveVariableHash variable_hash); /* loop unrolling of hash collision detection */


__device__ __inline__ CurveVariable getVariable(const CurveVariableHash variable_hash)
{
	const CurveVariableHash hash = variable_hash + d_namespace;
	for (unsigned int x=0; x< CURVE_MAX_VARIABLES; x++)
	{
		const CurveVariable i = ((hash + x) & (CURVE_MAX_VARIABLES-1));
		const CurveVariableHash h = d_hashes[i];
		if ( h == hash)
			return i;
	}
	return UNKNOWN_CURVE_VARIABLE;
}


/* header implementations */

__host__ CurveVariable curveGetVariable(CurveVariableHash variable_hash)
{
    unsigned int i, n;

    variable_hash += h_namespace;
    n = 0;
	i = (variable_hash) % CURVE_MAX_VARIABLES;

	while (h_hashes[i] != 0)
    {
        if (h_hashes[i] == variable_hash)
		{
			return i;
		}
        n += 1;
        if (n >= CURVE_MAX_VARIABLES)
		{
			break;
		}
        i += 1;
        if (i >= CURVE_MAX_VARIABLES)
        {
            i = 0;
        }
    }
	return UNKNOWN_CURVE_VARIABLE;
}





__host__ void curveInit(unsigned int default_vector_width)
{
	unsigned int *_d_hashes;
	float** _d_variables;
    int** _d_states;

	//set global vector size
	h_default_vector_width = default_vector_width;

	//namespace
	h_namespace = NAMESPACE_NONE;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_namespace, &h_namespace, sizeof(unsigned int)));

	//get a host pointer to d_hashes and d_variables
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_hashes, d_hashes));
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_variables, d_variables));
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_states, d_states));

	//set values of hash table to 0 on host and device
	memset(h_hashes, 0,  sizeof(unsigned int)*CURVE_MAX_VARIABLES);
	memset(h_states, 0,  sizeof(int)*CURVE_MAX_VARIABLES);

	//initialise data to 0 on device
	CUDA_SAFE_CALL(cudaMemset(_d_hashes, 0, sizeof(unsigned int)*CURVE_MAX_VARIABLES));
	CUDA_SAFE_CALL(cudaMemset(_d_variables, 0, sizeof(void*)*CURVE_MAX_VARIABLES));
	CUDA_SAFE_CALL(cudaMemset(_d_states, VARIABLE_DISABLED, sizeof(int)*CURVE_MAX_VARIABLES));

	curveClearErrors();
}

__host__ CurveVariable curveRegisterVariableByHash(CurveVariableHash variable_hash)
{
	return curveRegisterVariableByHashW(variable_hash, h_default_vector_width);
}

__host__ CurveVariable curveRegisterVariableByHashW(CurveVariableHash variable_hash, unsigned int vector_width)
{
	unsigned int i, n;
	unsigned int *_d_hashes;
	float** _d_variables;
	int** _d_states;

	n = 0;
	variable_hash += h_namespace;
	i = (variable_hash) % CURVE_MAX_VARIABLES;

	while (h_hashes[i] != 0)
	{
		n += 1;
		if (n >= CURVE_MAX_VARIABLES)
		{
			h_curve_error = CURVE_ERROR_TOO_MANY_VARIABLES;
			return UNKNOWN_CURVE_VARIABLE;
		}
		i += 1;
		if (i >= CURVE_MAX_VARIABLES)
		{
			i = 0;
		}
	}
	h_hashes[i] = variable_hash;

	//get a host pointer to d_hashes and d_variables
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_hashes, d_hashes));
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_variables, d_variables));
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_states, d_states));

	//copy hash to device
	CUDA_SAFE_CALL(cudaMemcpy(&_d_hashes[i], &h_hashes[i], sizeof(unsigned int), cudaMemcpyHostToDevice));

	//allocate the variable array on the device (with host pointer) and then copy to device
	CUDA_SAFE_CALL(cudaMalloc((void**) &h_d_variables[i], vector_width*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(&_d_variables[i], &h_d_variables[i], sizeof(float*), cudaMemcpyHostToDevice));

	//set the state to enabled
	h_states[i] = VARIABLE_ENABLED;
	CUDA_SAFE_CALL(cudaMemcpy(&_d_states[i], &h_states[i], sizeof(int), cudaMemcpyHostToDevice));

	//allocate the variable array on the host
	h_variables[i] = (float*)malloc(vector_width*sizeof(float));

	printf("Var name %s hash is %u at index %d with %d collisions\n", "todo", variable_hash, i, n);

	return i;
}

__host__ float curveGetFloatByHash(CurveVariableHash variable_hash, unsigned int index)
{
	CurveVariable cv = curveGetVariable(variable_hash);

	//error checking
	if (cv == UNKNOWN_CURVE_VARIABLE)
	{
		h_curve_error = CURVE_ERROR_UNKNOWN_VARIABLE;
		return 0.0f;
	}

	CUDA_SAFE_CALL(cudaMemcpy(&h_variables[cv][index], &h_d_variables[cv][index], sizeof(float), cudaMemcpyDeviceToHost));

	return h_variables[cv][index];
}

__host__ void curveSetFloatByHash(CurveVariableHash variable_hash, float value, unsigned int index)
{
	CurveVariable cv = curveGetVariable(variable_hash);

	//error checking
	if (cv == UNKNOWN_CURVE_VARIABLE)
	{
		h_curve_error = CURVE_ERROR_UNKNOWN_VARIABLE;
		return;
	}

	h_variables[cv][index] = value;
	CUDA_SAFE_CALL(cudaMemcpy(&h_d_variables[cv][index], &h_variables[cv][index], sizeof(float), cudaMemcpyHostToDevice));
}

__host__ void curveDisableVariableByHash(CurveVariableHash variable_hash)
{
	CurveVariable cv = curveGetVariable(variable_hash);
	int** _d_states;

	//error checking
	if (cv == UNKNOWN_CURVE_VARIABLE)
	{
		h_curve_error = CURVE_ERROR_UNKNOWN_VARIABLE;
		return;
	}

	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_states, d_states));
	h_states[cv] = VARIABLE_DISABLED;
	CUDA_SAFE_CALL(cudaMemcpy(&_d_states[cv], &h_states[cv], sizeof(int), cudaMemcpyHostToDevice));
}

__host__ void curveEnableVariableByHash(CurveVariableHash variable_hash)
{
	CurveVariable cv = curveGetVariable(variable_hash);
	int** _d_states;

	//error checking
	if (cv == UNKNOWN_CURVE_VARIABLE)
	{
		h_curve_error = CURVE_ERROR_UNKNOWN_VARIABLE;
		return;
	}

	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_states, d_states));
	h_states[cv] = VARIABLE_ENABLED;
	CUDA_SAFE_CALL(cudaMemcpy(&_d_states[cv], &h_states[cv], sizeof(int), cudaMemcpyHostToDevice));
}

__host__ void curveSetNamespaceByHash(CurveNamespaceHash namespace_hash)
{
	h_namespace = namespace_hash;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_namespace, &h_namespace, sizeof(unsigned int)));
}

__host__ void curveSetDefaultNamespace()
{
	h_namespace = NAMESPACE_NONE;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_namespace, &h_namespace, sizeof(unsigned int)));
}


__device__ float getFloatVariableByHash(const CurveVariableHash variable_hash, unsigned int index)
{
    CurveVariable cv;

    cv = getVariable(variable_hash);

    //error checking
    if (cv == UNKNOWN_CURVE_VARIABLE)
    {
    	d_curve_error = CURVE_DEVICE_ERROR_UNKNOWN_VARIABLE;
    	return 0.0f;
    }
    if(!d_states[cv])
    {
    	d_curve_error = CURVE_DEVICE_ERROR_VARIABLE_DISABLED;
    	return 0.0f;
    }

    return ((float*)d_variables[cv])[index];
}

__device__ void setFloatVariableByHash(const CurveVariableHash variable_hash, float variable, unsigned int index)
{
    CurveVariable cv;

    cv = getVariable(variable_hash);

    //error checking
    if (cv == UNKNOWN_CURVE_VARIABLE)
	{
		d_curve_error = CURVE_DEVICE_ERROR_UNKNOWN_VARIABLE;
		return;
	}
    if(!d_states[cv])
    {
    	d_curve_error = CURVE_DEVICE_ERROR_VARIABLE_DISABLED;
    	return;
    }

    ((float*)d_variables[cv])[index] = variable;
}


/* errors */
void __device__ curvePrintLastDeviceError(const char* file, const char* function, const int line){
	if (d_curve_error != CURVE_DEVICE_ERROR_NO_ERRORS)
	{
		printf("%s.%s.%d: cuRVE Device Error %d (%s)\n", file, function, line, (unsigned int)d_curve_error, curveGetDeviceErrorString(d_curve_error));
	}
}

void __host__ curvePrintLastHostError(const char* file, const char* function, const int line){
	if (h_curve_error != CURVE_ERROR_NO_ERRORS)
	{
		printf("%s.%s.%d: cuRVE Host Error %d (%s)\n", file, function, line, (unsigned int)h_curve_error, curveGetHostErrorString(h_curve_error));
	}
}

void __host__ curvePrintErrors(const char* file, const char* function, const int line){
	curveDeviceError d_curve_error_local;

	curvePrintLastHostError(file, function, line);

	//check device errors
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&d_curve_error_local, d_curve_error, sizeof(curveDeviceError)));
	if (d_curve_error_local != CURVE_DEVICE_ERROR_NO_ERRORS)
	{
		printf("%s.%s.%d: cuRVE Device Error %d (%s)\n", file, function, line, (unsigned int)d_curve_error_local, curveGetDeviceErrorString(d_curve_error_local));
	}
}

__device__ __host__ const char* curveGetDeviceErrorString(curveDeviceError e)
{
	switch (e){
		case(CURVE_DEVICE_ERROR_NO_ERRORS):
				return "No cuRVE errors";
		case(CURVE_DEVICE_ERROR_UNKNOWN_VARIABLE):
				return "Unknown cuRVE variable in current namespace";
		case(CURVE_DEVICE_ERROR_VARIABLE_DISABLED):
				return "cuRVE variable is disabled";
		default:
			return "Unspecified cuRVE error";
	}
}

__host__ const char* curveGetHostErrorString(curveHostError e)
{
	switch (e){
		case(CURVE_ERROR_NO_ERRORS):
				return "No cuRVE errors";
		case(CURVE_ERROR_UNKNOWN_VARIABLE):
				return "Unknown cuRVE variable";
		case(CURVE_ERROR_TOO_MANY_VARIABLES):
				return "Too many cuRVE variables";
		default:
			return "Unspecified cuRVE error";
	}
}

__device__ curveDeviceError curveGetLastDeviceError()
{
	return d_curve_error;
}

__host__ curveHostError curveGetLastHostError()
{
	return h_curve_error;
}


__host__ void curveClearErrors()
{
	curveDeviceError curve_error_none;

	curve_error_none = CURVE_DEVICE_ERROR_NO_ERRORS;
	h_curve_error  = CURVE_ERROR_NO_ERRORS;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_curve_error, &curve_error_none, sizeof(curveDeviceError)));

}

