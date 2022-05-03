# cuRVE Read-write Benchmark

cuRVE trades read/write performance for increased flexibility, but it is important to understand that cost. 

This benchmark compares use of cuRVE to read/write data from/to global memory, with minimal compute within the kernel to evaluate raw cuRVE performance, and compares it to performance of the same operations operating directly on global memory.

This done at a couple of scales, reading/writing from/to a few cuRVE addresses, or a larger number of memory addresses to compare the performance.  

This is to measure the overhead costs of accessing curve itself, so may not be representative of use in a more realistic scenario which does additional operations in between cuRVE reads/writes (see `abm-miniapp`).


There are `@todo` kernels in each of the implementations:

+ `someCoalescedWrites`
  + **Writes** data to global memory for a **small** number of independent regions of global memory, with **perfectly coalesced** writes.
+ `someCoalescedReads`
  + **Reads** data from global memory for a **small** number of independent regions of global memory, with **perfectly coalesced** reads.
+ `manyCoalescedWrites`
  + **Writes** data to global memory for a **larger** number of independent regions of global memory, with **perfectly coalesced** writes.
+ `manyCoalescedReads`
  + **Reads** data from global memory for a **larger** number of independent regions of global memory, with **perfectly coalesced** reads.
+ `someScatteredWrites`
  + **Writes** data to global memory for a **small** number of independent regions of global memory, with **Scattered** writes.
+ `someScatteredReads`
  + **Reads** data from global memory for a **small** number of independent regions of global memory, with **Scattered** reads.
+ `manyScatteredWrites`
  + **Writes** data to global memory for a **larger** number of independent regions of global memory, with **Scattered** writes.
+ `manyScatteredReads`
  + **Reads** data from global memory for a **larger** number of independent regions of global memory, with **Scattered** reads.
+ `someBroadcastReads`
  + **Reads** data from global memory for a **small** number of independent global memory addresses as a **broadcast** (all threads read the same address)
+ `manyBroadcastReads`
  + **Reads** data from global memory for a **larger** number of independent global memory addresses as a **broadcast** (all threads read the same address)

Each kernel is executed multiple times, to produce an average runtime. 

The average cuRVE 

## Compilation

> @todo

## Usage

> @todo