segmentedgpu
============

Segmented parallel primitives for GPUs.

segmentedgpu is a stripped-down version of [moderngpu v1.0](http://nvlabs.github.io/moderngpu) by Sean Baxter. The main differences are as follows:

* Only segmented and non-segmented reductions, scans and sorts are present (along any dependencies they may have.)
* A [StreamScan](https://dl.acm.org/citation.cfm?id=2442539) implementation has been added. Note that the StreamScan is suitable for compute capability 2.x devices, where the maximum block size is typically too small for larger problems. The default scan fails to operate correctly on 2.x devices when the block size exceeds 65535.
Also note that the current implementation does not fully conform to the CUDA memory model, and may in principle result in occasional errors (though none have been found on compute capability 2.0, 3.0 and 3.5 devices thus far for CUDA versions 6.0 through 7.5).
* A segmented scan has been added.
