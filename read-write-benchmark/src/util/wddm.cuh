#ifndef INCLUDE_UTIL_WDDM_CUH_
#define INCLUDE_UTIL_WDDM_CUH_

namespace util {
namespace wddm {

/**
 * Determine if a device is using the WDDM driver or not.
 * 
 * This impacts on the suitability of CUDAEventTimers amongst other things.
 * WDDM driver mode is only available under windows. 
 * 
 * @param deviceIndex the index of the device to be queried
 * @return boolean indicating if the WDDM driver is in use or not.
 */
bool deviceIsWDDM(int deviceIndex);

/**
 * Determine if the current device is using the WDDM driver or ot.
 * 
 * This impacts on the suitability of CUDAEventTimers amongst other things.
 * 
 * @return boolean indicating if the WDDM driver is in use or not.
 */
bool deviceIsWDDM();

}  // namespace wddm
}  // namespace util

#endif  // INCLUDE_UTIL_WDDM_CUH_
