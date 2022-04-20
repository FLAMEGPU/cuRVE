#ifndef __CURVE_RUNTIME_H__
#define __CURVE_RUNTIME_H__

/**
 * @file   curve.h
 * @Author Paul Richmond (p.richmond@sheffield.ac.uk) http://www.paulrichmond.staff.shef.ac.uk/
 * @date   January, 2014
 * @brief  Main cuRVE header file
 *
 * The main cuRVE header file for the CUDA Runtime Variable Environment (cuRVE)
 */


#include "cuda_runtime.h"

#define UNKNOWN_CURVE_VARIABLE 	-1						//!< value returned as a CurveVariable if an API function encounters an error

typedef int 					CurveVariable;			//!< Typedef for cuRVE variable handle
typedef unsigned int 			CurveVariableHash;		//!< Typedef for cuRVE variable name string hash
typedef unsigned int 			CurveNamespaceHash;		//!< Typedef for cuRVE variable namespace string hash

/**
 * Enumerator for GPU device error code which may be raised by CUDA kernels
 */
enum curveDeviceError
{
	CURVE_DEVICE_ERROR_NO_ERRORS,				//!< No errors raised on the device
	CURVE_DEVICE_ERROR_UNKNOWN_VARIABLE,		//!< A function has requested an unknown variable or a variable not registered in the current namespace
	CURVE_DEVICE_ERROR_VARIABLE_DISABLED		//!< A function has requested a variable which is disabled
};

/**
 * Enumerator for cuRVE host error codes which may be raised by cuRVE API function calls
 */
enum curveHostError
{
	CURVE_ERROR_NO_ERRORS,						//!< No errors raised by host API functions
	CURVE_ERROR_UNKNOWN_VARIABLE,				//!< A host API function has requested an unknown variable or a variable not registered in the current namespace
	CURVE_ERROR_TOO_MANY_VARIABLES				//!< The maximum number of curve variables has been reached
};


/* TEMPLATE HASHING FUNCTIONS */

/** @brief Non terminal template structure has function for a constant char array
 * 	Use of template meta-programming ensures the compiler can evaluate string hashes at compile time. This reduces constant string variable names to a single 32 bit value. Hashing is based on 'Quasi Compile Time String Hashing' at http://www.altdevblogaday.com/2011/10/27/quasi-compile-time-string-hashing/
 * 	Code uses compilation flags for both the host and the CUDA device.
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N, unsigned int I> struct CurveStringHash
{
	__device__ __host__ inline static CurveVariableHash Hash(const char (&str)[N])
	{
		return (CurveStringHash<N, I-1>::Hash(str) ^ str[I-1])*16777619u;
	}
};
/** @brief Terminal template structure has function for a constant char array
 * 	Function within a template structure allows partial template specialisation for terminal case.
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N> struct CurveStringHash<N, 1>
{
	__device__ __host__ inline static CurveVariableHash Hash(const char (&str)[N])
	{
	    return (2166136261u ^ str[0])*16777619u;
	}
};
/** @brief Main cuRVE variable hashing function
 *  Calls recursive hashing functions
 *  @return a 32 bit cuRVE string variable hash.
 */
template <unsigned int N> __device__ __host__ inline static CurveVariableHash curveVariableHash(const char (&str)[N])
{
	return CurveStringHash<N, N>::Hash(str);
}


/* CURVE HOST API FUNCTIONS */

/** @brief cuRVE initialisation function
 *  Main initialisation function. Must be called before any other cuRVE API calls.
 *  @param default_vector_width	The default vector width size of any variables declared with curveRegisterVariable.
 */
__host__ void curveInit(unsigned int default_vector_width);

/** @brief Function for getting a handle to a cuRVE variable from a variable string hash
 * 	Function performs hash collision avoidance using linear probing.
 *  @param variable_hash A cuRVE variable string hash from curveVariableHash.
 *  @return CurveVariable Handle for the cuRVE variable.
 */
__host__ CurveVariable curveGetVariable(CurveVariableHash variable_hash);


/** @brief Function for registering a variable by a CurveVariableHash
 * 	Registers a variable by insertion in a hash table. Recommend using the provided curveRegisterVariable template function. The default vector width (set by curveInit) will be used.
 *  @param variable_hash A cuRVE variable string hash from curveVariableHash.
 *  @return CurveVariable Handle of registered variable or UNKNOWN_CURVE_VARIABLE if an error is encountered.
 */
__host__ CurveVariable curveRegisterVariableByHash(CurveVariableHash variable_hash);

/** @brief Template function for registering a constant string
 * 	Registers a constant string variable name by hashing and then inserting into a hash table.
 *  @param variableName A constant char array (C string) variable name.
 *  @return CurveVariable Handle of registered variable or UNKNOWN_CURVE_VARIABLE if an error is encountered.
 */
template <unsigned int N> __host__ CurveVariable curveRegisterVariable(const char (&variableName)[N]){
	CurveVariableHash variable_hash = curveVariableHash(variableName);
	return curveRegisterVariableByHash(variable_hash);
}


/** @brief Function for registering a variable by a CurveVariableHash with a specified vector width
 * 	Registers a variable by insertion in a hash table using the specified vector width. Recommend using the provided curveRegisterVariableW template function.
 *  @param variable_hash A cuRVE variable string hash from curveVariableHash.
 *  @return CurveVariable Handle of registered variable or UNKNOWN_CURVE_VARIABLE if an error is encountered.
 */
__host__ CurveVariable curveRegisterVariableByHashW(CurveVariableHash variable_hash, unsigned int vector_width);

/** @brief Template function for registering a constant string with a specified vector width
 * 	Registers a constant string variable name with the specified vector width by hashing and then inserting into a hash table.
 *  @param variableName A constant char array (C string) variable name.
 *  @return CurveVariable Handle of registered variable or UNKNOWN_CURVE_VARIABLE if an error is encountered.
 */
template <unsigned int N> __host__ CurveVariable curveRegisterVariableW(const char (&variableName)[N], unsigned int vector_width){
	CurveVariableHash variable_hash = curveVariableHash(variableName);
	return curveRegisterVariableByHashW(variable_hash, vector_width);
}


/** @brief Function for getting a single float value from a CurveVariableHash
 * 	Returns a single value from a curveVariableHash using the given index position.
 *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
 *  @param index The index of the variable in the named variable vector.
 *  @return float A float value at the given index for the variable with the provided hash. Will return 0.0f if an error is raised.
 */
__host__ float curveGetFloatByHash(CurveVariableHash variable_hash, unsigned int index);

/** @brief Template function for getting a single float value from a constant string variable name
 * 	Returns a single value from a constant string expression using the given index position
 *  @param variableName A constant string variable name which must have been registered as a cuRVE variable.
 *  @param index The index of the variable in the named variable vector
 *  @return float A float value at the given index for the variable with the provided hash. Will return 0.0f if an error is raised.
 */
template <unsigned int N> __host__ float curveGetFloat(const char (&variableName)[N], unsigned int index){
	CurveVariableHash variable_hash = curveVariableHash(variableName);
	return curveGetFloatByHash(variable_hash, index);
}


/** @brief Function for setting a single float value from a CurveVariableHash
 * 	Sets a single value from a curveVariableHash using the given index position.
 *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
 *  @param index The index of the variable in the named variable vector.
 *  @param value The float value to set at the given index.
 */
__host__ void curveSetFloatByHash(CurveVariableHash variable_hash, float value, unsigned int index);

/** @brief Template function for getting a setting float value from a constant string variable name
 * 	Returns a single value from a constant string expression using the given index position
 *  @param variableName A constant string variable name which must have been registered as a curve variable.
 *  @param index The index of the variable in the named variable vector
 *  @param value The float value to set at the given index.
 */
template <unsigned int N> __host__ void curveSetFloat(const char (&variableName)[N], float value, unsigned int index){
	CurveVariableHash variable_hash = curveVariableHash(variableName);
	curveSetFloatByHash(variable_hash, value, index);
}


/** @brief Function for disabling access to a cuRVE variable from a CurveVariableHash
 * 	Disables device access to the cuRVE variable. Does not disable host access.
 *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
 */
__host__ void curveDisableVariableByHash(CurveVariableHash variable_hash);

/** @brief Template function for disabling access to a cuRVE variable from a constant string variable name
 * 	Disables device access to the cuRVE variable. Does not disable host access.
 *  @param variableName A constant string variable name which must have been registered as a curve variable
 */
template <unsigned int N> __host__ void curveDisableVariable(const char (&variableName)[N]){
	CurveVariableHash variable_hash = curveVariableHash(variableName);
	curveDisableVariableByHash(variable_hash);
}


/** @brief Function for enabling access to a cuRVE variable from a CurveVariableHash
 * 	Enables device access to the cuRVE variable.
 *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
 */
__host__ void curveEnableVariableByHash(CurveVariableHash variable_hash);

/** @brief Template function for enabling access to a cuRVE variable from a constant string variable name
 * 	Enables device access to the cuRVE variable.
 *  @param variableName A constant string variable name which must have been registered as a curve variable
 */
template <unsigned int N> __host__ void curveEnableVariable(const char (&variableName)[N]){
	CurveVariableHash variable_hash = curveVariableHash(variableName);
	curveEnableVariableByHash(variable_hash);
}


/** @brief Function changes the current namespace from a CurveNamespaceHash
 * 	Changing the namespace will affect both the host and device.
 *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
 */
__host__ void curveSetNamespaceByHash(CurveNamespaceHash variable_hash);

/** @brief Template function changes the current namespace using a constant string namespace name
 * 	Changing the namespace will affect both the host and device.
 *  @param namespaceName A constant string variable name which must have been registered as a curve variable
 */
template <unsigned int N> __host__ void curveSetNamespace(const char (&namespaceName)[N]){
	CurveNamespaceHash namespace_hash = curveVariableHash(namespaceName);
	curveSetNamespaceByHash(namespace_hash);
}

/** @brief Function changes the current namespace to the default empty namespace
 * 	Changing the namespace will affect both the host and device.
 */
__host__ void curveSetDefaultNamespace();




/* DEVICE API FUNCTIONS */

/** @brief Device function for getting a single float value from a CurveVariableHash at a given index
 * 	Returns a single value from a curveVariableHash using the given index position.
 *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
 *  @param index The index of the variable in the named variable vector.
 *  @return float A float value at the given index for the variable with the provided hash. Will return 0.0f if an error is raised.
 */
extern __device__ float getFloatVariableByHash(const CurveVariableHash variable_hash, unsigned int index);

/** @brief Template device function for getting a single float value from a constant string variable name
 * 	Returns a single value from a constant string expression using the given index position
 *  @param variableName A constant string variable name which must have been registered as a cuRVE variable.
 *  @param index The index of the variable in the named variable vector
 *  @return float A float value at the given index for the variable with the provided hash. Will return 0.0f if an error is raised.
 */
template <unsigned int N> __device__ float getFloatVariable(const char (&variableName)[N], unsigned int index)
{
	CurveVariableHash variable_hash = curveVariableHash(variableName);
	return getFloatVariableByHash(variable_hash, index);
}


/** @brief Device function for setting a single float value from a CurveVariableHash
 * 	Sets a single value from a curveVariableHash using the given index position.
 *  @param variable_hash A cuRVE variable string hash from CurveVariableHash.
 *  @param index The index of the variable in the named variable vector.
 *  @param value The float value to set at the given index.
 */
extern __device__ void setFloatVariableByHash(const CurveVariableHash variable_hash, float variable, unsigned int index);

/** @brief Device template function for getting a setting float value from a constant string variable name
 * 	Returns a single value from a constant string expression using the given index position
 *  @param variableName A constant string variable name which must have been registered as a curve variable.
 *  @param index The index of the variable in the named variable vector
 *  @param value The float value to set at the given index.
 */
template <unsigned int N> __device__ void setFloatVariable(const char (&variableName)[N], float variable, unsigned int index){
	CurveVariableHash variable_hash = curveVariableHash(variableName);
	setFloatVariableByHash(variable_hash, variable, index);
}

/* ERROR CHECKING API FUNCTIONS */

/** @brief Device function for printing the last device error
 * 	Prints the last device error using the provided source location information. The preferred method for printing is to use the curveReportLastDeviceError macro which inserts source location information.
 *  @param file A constant string filename.
 *  @param function A constant string function name.
 *  @param line A constant integer line number.
 */
__device__ void curvePrintLastDeviceError(const char* file, const char* function, const int line);

#define curveReportLastDeviceError() { curvePrintLastDeviceError(__FILE__, __FUNCTION__, __LINE__); }	//! Prints the last reported device error using the file, function and line number of the call to this macro

/** @brief Host API function for printing the last host error
 * 	Prints the last host API error using the provided source location information. The preferred method for printing is to use the curveReportLastHostError macro which inserts source location information.
 *  @param file A constant string filename.
 *  @param function A constant string function name.
 *  @param line A constant integer line number.
 */
void __host__ curvePrintLastHostError(const char* file, const char* function, const int line);

#define curveReportLastHostError() { curvePrintLastHostError(__FILE__, __FUNCTION__, __LINE__); }		//! Prints the last reported host API error using the file, function and line number of the call to this macro

/** @brief Host API function for printing the last host or device error
 * 	Prints the last device or host API error (or both) using the provided source location information. The preferred method for printing is to use the curveReportErrors macro which inserts source location information.
 *  @param file A constant string filename.
 *  @param function A constant string function name.
 *  @param line A constant integer line number.
 */
void __host__ curvePrintErrors(const char* file, const char* function, const int line);

#define curveReportErrors() { curvePrintErrors(__FILE__, __FUNCTION__, __LINE__); }						//! Prints the last reported device or host API error using the file, function and line number of the call to this macro

/** @brief Device API function for returning a constant string error description
 * 	Returns an error description given a curveDeviceError error code.
 *  @param error_code A curveDeviceError error code.
 *  @return constant A string error description
 */
__device__ __host__ const char*  curveGetDeviceErrorString(curveDeviceError error_code);

/** @brief Host API function for returning a constant string error description
 * 	Returns an error description given a curveHostError error code.
 *  @param error_code A curveHostError error code.
 *  @return constant A string error description
 */
__host__ const char*  curveGetHostErrorString(curveHostError error_code);

/** @brief Device API function for returning the last reported error code
 *  @return A curveDeviceError error code
 */
__device__ curveDeviceError curveGetLastDeviceError();

/** @brief Host API function for returning the last reported error code
 *  @return A curveHostError error code
 */
__host__ curveHostError curveGetLastHostError();

/** @brief Host API function for clearing both the device and host error codes
 */
__host__ void curveClearErrors();

#endif //__CURVE_RUNTIME_H__
