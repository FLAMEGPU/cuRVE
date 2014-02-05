cuRVE The CUDA Runtime Variable Environment
=====

=== Overview ===

The CUDA Runtime Variable Environment (curRVE) is a library which provides key-value memory management and access for CUDA global device memory.

GPU device memory can be registered and values set in host code using a constant string expression key via the the cuRVE API. For example the following initialises and sets the value of three cuRVE variables.

curveInit(VECTOR_ELEMENTS);
curveRegisterVariable("a");
curveRegisterVariable("b");
curveRegisterVariable("c");

for (int i=0; i<VECTOR_ELEMENTS; i++){
    float a = rand()/(float)RAND_MAX;
    float b = rand()/(float)RAND_MAX;
    curveSetFloat("a", i, a);
    curveSetFloat("b", i, b);
	}
	
A corresponding kernel can be defined using the cuRVE variable access functions as follows;
	
	__global__ void vectorAdd()
{
    float a, b, c;
    
    unsigned int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
     
    a = getFloatVariable("a", idx);
    b = getFloatVariable("b", idx);
    c = a + b;
    setFloatVariable("c", c, idx);
}

=== Namespaces ===

Namespaces allow a cuRVE variable to have a limited scope. Similarly, variable names can be re-used under different namespaces. The cuRVE namespace can be changed using  an API call as follows;

curveChangeNamespace("vector_addition_example");

Namespaces can be used to limit a variable to a particular CUDA kernel (or set of kernels). Further restrictions on variable access may be placed by enabling or disabling variable access as follows;

curveDisableVariable("a");
curveEnableVariable("b");

=== Error Checking ===

Errors can occur on the device or host and error codes can be obtained by using the following API functions;

curveGetLastHostError();    // Host API function which gets the last host API error code
curveGetLastDeviceError();  // Device API function which gets the last device API error code

Formatted errors can be output using the following API calls which will use the current source file, function name and line number;

curveReportErrors();      //Host API function outputs any host or device errors to std:out
curveReportHostError();   //Host API function outputs the last host API error
curveReportDeviceError(); //Device API function outputs the last device API error

=== Documentation ===

The cuRVE header file is commented using doxygen format comments.
