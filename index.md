# ACCL: the Adaptive Collective Communication Library

ACCL (pronounced like [Achill](https://goo.gl/maps/4e3vGfa5BsT5s3vm9), the Irish island) is a MPI-like communication library designed for the needs of FPGA-accelerated applications. Similar to NVidia's [NCCL](https://github.com/NVIDIA/nccl) and AMD's [RCCL](https://github.com/ROCmSoftwarePlatform/rccl), ACCL provides a small but highly FPGA-optimized subset of MPI collectives. With ACCL, applications can move data directly between FPGA local memories, utilizing the 100 Gbps Ethernet ports on Alveo FPGA accelerator cards, and avoiding copies to host memory or transfers over host NICs. This makes it similar to Nvidia GPUDirect, except with ACCL a single card serves for both networking and compute.

## Why MPI, and how much of it?

The Message Passing Interface ([MPI](http://mpi-forum.org/)), is a standardized and portable message-passing system designed to function on a wide variety of parallel computers. Its wide availability, universal support and high performance make it the lingua franca of distributed computing and a natural API to follow when implementing specialized distributed computation frameworks such as ACCL. ACCL implements the send and receive MPI primitives and seven of the most common collectives:
- send and recv
- broadcast
- gather and allgather
- scatter and reduce-scatter
- reduce and allreduce

## ACCL Structure

ACCL is a combination of software running on the host CPU, FPGA data-moving hardware, and control firmware executing on a FPGA-embedded microcontroller. Here is a high level overview of the ACCL structure:

![schematic](images/ccl_kernels.svg)

In the FPGA, ACCL features a collectives offload engine (`CCLO`) and one or more network protocol offload engines (`POE`), each of which is implemented as a stand-alone Vitis kernel. The `CCLO` implements the collectives by orchestrating data movement between host, FPGA memory and POEs, and is described in more detail in the [Hardware](./kernel.md) page. The protocol offload engines implements the full network stack up to `UDP` and `TCP/IP` respectively and connect directly to Ethernet ports, e.g. through Alveo Gigabit Transceivers and `QSFP28` ports. The host communicates with the CCLO over `PCIe` and `Xilinx XDMA`, but this complexity is hidden by XRT and our drivers, as described in the [API](./api.md) page. 
