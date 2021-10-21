# ACCL: the Adaptive Collective Communication Library

ACCL (pronounced like [Achill](https://goo.gl/maps/4e3vGfa5BsT5s3vm9), the Irish island) is a MPI-like communication library designed for the needs of FPGA-accelerated applications. Similar to NVidia's [NCCL](https://github.com/NVIDIA/nccl) and AMD's [RCCL](https://github.com/ROCmSoftwarePlatform/rccl), ACCL provides a small but highly FPGA-optimized subset of MPI collectives. With ACCL, applications can move data directly between FPGA local memories, utilizing the 100 Gbps Ethernet ports on Alveo  FPGA accelerator cards, and avoiding copies to host memory or transfers over host NICs. This makes it similar to Nvidia GPUDirect, except with ACCL a single card serves for both networking and compute.

## Why MPI, and how much of it?

The Message Passing Interface ([MPI](http://mpi-forum.org/)), is a standardized and portable message-passing system designed to function on a wide variety of parallel computers. Its wide availability, universal support and high performance make it the lingua franca of distributed computing and a natural API to follow when implementing specialized distributed computation frameworks such as ACCL. ACCL implements send and receive MPI primitives and seven of the most common collectives:
- send and recv
- broadcast
- gather and allgather
- scatter and reduce-scatter
- reduce and allreduce

## Next Sections

[API](./api.md)

[Hardware](./kernel.md)

[Collectives](./collectives.md)

[Building](./build.md)

[Debugging](./debug.md)