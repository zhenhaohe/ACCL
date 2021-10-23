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
from accl import accl

# receive binfile, list of ranks as inputs
...
# configure ACCL driver and the CCLO kernel in the FPGA
accl_inst = accl(xclbin, ranks, MPI.COMM_WORLD.Get_rank(), protocol="TCP")
````

This code, which would be executed with `mpirun` on each node of a cluster of Alveo-equipped nodes, starts by programming the node FPGAs and getting unique process ranks from the MPI4Py module. Subsequently, the ACCL driver is configured by calling its constructor which, in sequence, allocates message buffers in FPGA local memory, configures the IPs and associated rank numbers of known FPGA peers, then opens ports and TCP sessions to the peers. 

At this point the FPGAs are ready to exchange data. We first communicate through the send and recv primitives:

````
txb=allocate((16384,), target=accl_inst.devicemem)
rxb=allocate((16384,), target=accl_inst.devicemem)

# utilize primitives to exchange data
if rank==0:
    accl_inst.recv(rxb, src=1, tag=1, to_fpga=False, async=False)
elif rank==1:
    accl_inst.send(txb, dst=0, tag=1, from_fpga=True, async=False)
````

In this example rank 1 sends the contents of its `txb` buffer, which is located in the FPGA local memory - note `from_fpga=True` - into the `rxb` buffer on rank 0, in host memory, hence `to_fpga=False`. Each ACCL function has one or both of the `to_fpga`/`from_fpga` arguments, which allows users  to specify whether buffer arguments reside in the host or in FPGA off-chip memory. When required, ACCL moves data between host and FPGA. Similarly, each ACCL function has a `async` argument which determines the blocking behaviour of the call. Here, both calls are synchronous, i.e. blocking.

Any communication pattern between multiple ranks, including collectives, can be acomplished by properly assembling `send`/`recv` pairs. However, note that only the ACCL functions are offloaded to hardware, and any control executes on the host. Therefore, when possible, it is more efficient to utilize ACCL collectives - we exemplify allreduce and allgather here:

````
# execute offloaded collectives
ch = accl_inst.allreduce(txb, rxb, count=256, from_fpga=False, to_fpga=True, async=True)
accl_inst.allgather(txb, rxb, count=256, from_fpga=False, to_fpga=True, waitfor=[ch], async=False)
````

Here we perform a FPGA-orchestrated `allreduce` of data in `txb`, initially in host memory, with results in `rxb` in FPGA memory. The allreduce is executed asynchronously, i.e. nonblocking, and returns a handle which can be utilized as a dependency for the subsequent call to `allgather`, which will wait until the allreduce has finished then execute. We call this feature `chaining`. The dependencies are resolved by the Xilinx Embedded Run-Time in the FPGA, reducing the latency of executing long call graphs.

Finally, once the application is finished we reset the CCLO, clearing all internal configuration and freeing message buffers:

````
# release FPGA memory, reset CCLO
accl_inst.deinit()
````

## API Reference

### Primitives API

````
send(comm_id, srcbuf, dst, tag=TAG_ANY, from_fpga=False, run_async=False, waitfor=[])
````
ACCL equivalent for MPI `send`. Sends a buffer to another over the network. Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| comm_id | Numerical ID of the communicator in which the transmission is performed |
| srcbuf | Source Pynq buffer |
| dst | Rank ID of the destination |
| tag | Numerical tag, used for pairing sender with receiver |
| from_fpga | Signals if data is on FPGA before transmission. If not, ACCL will copy from host to FPGA before sending |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

````
recv(comm_id, dstbuf, src, tag=TAG_ANY, to_fpga=False, run_async=False, waitfor=[])
````
ACCL equivalent for MPI `recv`. Receives a buffer from another node over the network. Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| comm_id | Numerical ID of the communicator in which the transmission is performed |
| dstbuf | Destination Pynq buffer |
| src | Rank ID of the source |
| tag | Numerical tag, used for pairing sender with receiver |
| to_fpga | Signals if data should remain on FPGA after reception. If not, ACCL will copy from FPGA to host before returning |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

````
copy(srcbuf, dstbuf,  from_fpga=False,  to_fpga=False, run_async=False, waitfor=[])
````
Local copy between `srcbuf` and `dstbuf`. Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| srcbuf | Source Pynq buffer |
| dstbuf | Destination Pynq buffer |
| from_fpga | Signals if data is on FPGA before copy. If not, ACCL will copy the source data from host to FPGA before copying to destination |
| to_fpga | Signals if copied data should remain on FPGA. If not, ACCL will copy destination data from FPGA to host before returning |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

````
combine(func, val1, val2, result, val1_from_fpga=False, val2_from_fpga=False, to_fpga=False, run_async=False, waitfor=[])
````
Apply a reduction function locally to buffers `val1` and `val2`. Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| val1 | Source Pynq buffer |
| val2 | Source Pynq buffer |
| result | Source Pynq buffer |
| val1_from_fpga | Signals if val1 is on FPGA before applying the reduction. If not, ACCL will copy val1 from host to FPGA |
| val2_from_fpga | Signals if val2 is on FPGA before applying the reduction. If not, ACCL will copy val2 from host to FPGA |
| to_fpga | Signals if result should remain on FPGA. If not, ACCL will copy from FPGA to host before returning |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

### Collectives API

````
bcast(comm_id, buf, root, from_fpga=False, to_fpga=False, run_async=False, waitfor=[])
````
ACCL equivalent for MPI broadcast. A root node sends a buffer to all others over the network. Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| comm_id | Numerical ID of the communicator in which the transmission is performed |
| buf | Source Pynq buffer on the root, or destination Pynq buffer elsewhere |
| root | Rank ID of the root |
| from_fpga | Signals if data is on FPGA before transmission. If not, ACCL will copy from host to FPGA before sending |
| to_fpga | Signals if result should remain on FPGA. If not, ACCL will copy from FPGA to host before returning |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

````
scatter(comm_id, sbuf, rbuf, count, root, from_fpga=False, to_fpga=False, run_async=False, waitfor=[])
````
ACCL equivalent for MPI scatter. A root node sends segments of a source buffer to all nodes. Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| comm_id | Numerical ID of the communicator in which the transmission is performed |
| sbuf | Source Pynq buffer on the root, unused elsewhere |
| rbuf | Destination Pynq buffer |
| count | Number of elements to transmit |
| root | Rank ID of the root |
| from_fpga | Signals if data is on FPGA before transmission. If not, ACCL will copy from host to FPGA before sending |
| to_fpga | Signals if result should remain on FPGA. If not, ACCL will copy from FPGA to host before returning |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

````
gather(comm_id, sbuf, rbuf, count, root, from_fpga=False, to_fpga=False, run_async=False, waitfor=[])
````
ACCL equivalent for MPI gather. A root node receives segments of a destination buffer from all nodes. Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| comm_id | Numerical ID of the communicator in which the transmission is performed |
| sbuf | Source Pynq buffer |
| rbuf | Destination Pynq buffer on the root, unused elsewhere |
| count | Number of elements to transmit |
| root | Rank ID of the root |
| from_fpga | Signals if data is on FPGA before transmission. If not, ACCL will copy from host to FPGA before sending |
| to_fpga | Signals if result should remain on FPGA. If not, ACCL will copy from FPGA to host before returning |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

````
allgather(comm_id, sbuf, rbuf, count, from_fpga=False, to_fpga=False, run_async=False, waitfor=[])
````
ACCL equivalent for MPI allgather. Each node sends its send buffer and receives the concatenation of buffers sent by all nodes (including itself). Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| comm_id | Numerical ID of the communicator in which the transmission is performed |
| sbuf | Source Pynq buffer |
| rbuf | Destination Pynq buffer |
| count | Number of elements to transmit |
| from_fpga | Signals if data is on FPGA before transmission. If not, ACCL will copy from host to FPGA before sending |
| to_fpga | Signals if result should remain on FPGA. If not, ACCL will copy from FPGA to host before returning |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

````
reduce(comm_id, sbuf, rbuf, count, root, func, from_fpga=False, to_fpga=False, run_async=False, waitfor=[])
````
ACCL equivalent for MPI reduce. Each node sends its send buffer and the root node receives and combines these buffers using the specifie function. Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| comm_id | Numerical ID of the communicator in which the transmission is performed |
| sbuf | Source Pynq buffer |
| rbuf | Destination Pynq buffer in the root, otherwise unused |
| count | Number of elements to transmit |
| root | Rank ID of the root |
| func | Function ID in `ACCLReduceFunctions` |
| from_fpga | Signals if data is on FPGA before transmission. If not, ACCL will copy from host to FPGA before sending |
| to_fpga | Signals if result should remain on FPGA. If not, ACCL will copy from FPGA to host before returning |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

````
allreduce(comm_id, sbuf, rbuf, count, func, from_fpga=False, to_fpga=False, run_async=False, waitfor=[])
````
ACCL equivalent for MPI all-reduce. Similar to reduce, but each node gets the result of the reduction. Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| comm_id | Numerical ID of the communicator in which the transmission is performed |
| sbuf | Source Pynq buffer |
| rbuf | Destination Pynq buffer |
| count | Number of elements to transmit |
| func | Function ID in `ACCLReduceFunctions` |
| from_fpga | Signals if data is on FPGA before transmission. If not, ACCL will copy from host to FPGA before sending |
| to_fpga | Signals if result should remain on FPGA. If not, ACCL will copy from FPGA to host before returning |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

````
reduce_scatter(comm_id, sbuf, rbuf, count, func, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
````
ACCL equivalent for MPI scatter-reduce. Equivalent to a reduce followed by a scatter. Returns a call handle if called asynchronously.
| Argument  | Description |
| ------------- | ------------- |
| comm_id | Numerical ID of the communicator in which the transmission is performed |
| sbuf | Source Pynq buffer |
| rbuf | Destination Pynq buffer |
| count | Number of elements to transmit |
| func | Function ID in `ACCLReduceFunctions` |
| from_fpga | Signals if data is on FPGA before transmission. If not, ACCL will copy from host to FPGA before sending |
| to_fpga | Signals if result should remain on FPGA. If not, ACCL will copy from FPGA to host before returning |
| run_async | Controls asynchronous execution |
| waitfor | List of call dependencies |

### Configuration API

````
accl(xclbin, ranks, local_rank, protocol, board_idx, nbufs, bufsize, mem, arith_config)
````
The ACCL constructor, which allocates message buffers for the receive pipeline of the CCLO and for the TCP POE if requested, configures communicators, and initializes the ACCL kernel. Does not return anything, but as side-effect, it sets all ACCL fields.
| Argument  | Description |
| ------------- | ------------- |
| xclbin  | Path to XCLBIN file, which will be programmed into the FPGA |
| ranks | List of dictionaries of format {"ip": <int>, "port": <int>} representing IPs and ports of communicator peers |
| local_rank | Index of the rank where the function is being called. Typically obtained from MPI_COMM_WORLD if application launched from `mpirun` |
| protocol | One of "TCP" or "UDP", selecting between the two protocols as backend, if both exist |
| board_idx | Which Alveo board to utilize, if there are multiple in the system |
| nbufs  | Number of receive buffers to allocate |
| bufsize | Size of each allocated receive buffer |
| mem | list of (lists of) Pynq memory objects to be utilized for housekeeping. If set to `None`, ACCL tries to self-configure |
| arith_config | A list of instances of `ACCLArithConfig` which describe the arithmetic and compression plugins to ACCL |
