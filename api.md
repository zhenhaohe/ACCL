# Using ACCL

## Programmer's Perspective

From the perspective of the application developer, ACCL is a lightweight Python or C++ library providing three classes of functions:
- configuration functions, which set up the FPGA-side of ACCL in preparation for communication
- primitives, which are simple operations offloaded to FPGA hardware, including send and receive
- collectives, which orchestrate a series of primitives to achieve more complex communication patterns

We can see these APIs in action in the following code snippets: 
````
from pynq import Overlay, allocate
from mpi4py import MPI
#receive binfile, ranks_dict as inputs
ol = Overlay(binfile)
accl = ol.cclo
rank = MPI.COMM_WORLD.Get_rank()
bs = 16384

# configure ACCL driver and the CCLO kernel in the FPGA 
accl.setup_rx_buffers(nbufs=16, bufsize=bs, devicemem=ol.bank0)
accl.configure_communicator(ranks_dict, rank)
accl.open_port() 
accl.open_con()
````

This code, which would be executed with `mpirun` on each node of a cluster of Alveo-equipped nodes, starts by programming the node FPGAs and getting unique process ranks from the MPI4Py module. Subsequently, the ACCL driver is configured by calling four functions which, in sequence, allocate message buffers in FPGA local memory, configure the IPs and associated rank numbers of known FPGA peers, then open ports and TCP session to the peers. 

At this point the FPGAs are ready to exchange data. We first communicate through the send and recv primitives:

````
txb=allocate((bs,), target=ol.bank0)
rxb=allocate((bs,), target=ol.bank0)

# utilize primitives to exchange data
if rank==0:
    accl.recv(rxb, src=1, tag=1, to_fpga=False, async=False)
elif rank==1:
    accl.send(txb, dst=0, tag=1, from_fpga=True, async=False)
````

In this example rank 1 sends the contents of its `txb` buffer, which is located in the FPGA local memory - note `from_fpga=True` - into the `rxb` buffer on rank 0, in host memory, hence `to_fpga=False`. Each ACCL function has one or both of the `to_fpga`/`from_fpga` arguments, which allows users  to specify whether buffer arguments reside in the host or in FPGA off-chip memory. When required, ACCL moves data between host and FPGA. Similarly, each ACCL function has a `async` argument which determines the blocking behaviour of the call. Here, both calls are synchronous, i.e. blocking.

Any communication pattern between multiple ranks, including collectives, can be acomplished by properly assembling `send`/`recv` pairs. However, note that only the ACCL functions are offloaded to hardware, and any control executes on the host. Therefore, when possible, it is more efficient to utilize ACCL collectives - we exemplify allreduce and allgather here:

````
# execute offloaded collectives
ch = accl.allreduce(txb, rxb, count=256, from_fpga=False, to_fpga=True, async=True)
accl.allgather(txb, rxb, count=256, from_fpga=False, to_fpga=True, waitfor=[ch], async=False)

# release FPGA memory, reset CCLO
accl.deinit()
````

Here we perform a FPGA-orchestrated `allreduce` of data in `txb`, initially in host memory, with results in `rxb` in FPGA memory. The allreduce is executed asynchronously, i.e. nonblocking, and returns a handle which can be utilized as a dependency for the subsequent call to `allgather`, which will wait until the allreduce has finished then execute. We call this feature `chaining`. The dependencies are resolved by the Xilinx Embedded Run-Time in the FPGA, reducing the latency of executing long call graphs.

