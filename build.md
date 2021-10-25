
# Getting Started with ACCL

There are three steps to getting an ACCL system up and running: 
1. compile the ACCL CCL Offload kernel, the POE(s), and optionally your compute kernels
2. link the kernels into a bitstream
3. build and run host code

We provide demo scripts to compile bitstreams for Alveo U250 and U280 boards against the TCP POE, as well as functional verification and benchmarking host code to allow ACCL evaluation.

## Running a Single-Board Demo
We provide an example design with three instances of the ACCL hardware in a single board, connected together through a simulated swithc, which allows experimenting with ACCL without 100G Ethernet infrastructure. We'll need Vivado 2020.1 and 2020.2, and an Alveo U250 or U280. We currently support three Alveo shells as targets for demo designs:
- `xilinx_u250_gen3x16_xdma_3_1_202020_1`
- `xilinx_u250_xdma_201830_2`
- `xilinx_u280_xdma_201920_3`

We'll first clone the ACCL repo, build the TCP POE with Vivado 2020.1, then build and link the rest of the FPGA design using Vivado 2020.2:
````
git clone https://github.com/Xilinx/ACCL.git
cd accl/demo/build
<source Vivado 2020.1 settings script>
make PLATFORM=<shell name> tcp_stack_ips
<source Vivado 2020.2 settings script>
make PLATFORM=<shell name> MODE=tri FREQUENCY=100 
````
The build should complete in about a day, depending on your compute hardware, and will produce `link_tri_eth_none_debug_none_<shell name>/ccl_offload.xclbin`. Next, let's run a validation test against this bitstream. Since there are three instances of the ACCL hardware in this design, the host software emulates three nodes. The following must be executed on a Alveo-equipped node (make sure to have the shell specified in the build process) with XRT and Pynq installed.
````
cd ../host
python test.py --xclbin ../build/link_tri_eth_none_debug_none_<shell name>/ccl_offload.xclbin --all --benchmark
````
The test script will program the bitstream into the alveo, run tests against each of the ACCL primitives and collectives, and then do performance evaluation.

## Building Custom Systems with ACCL
Most users will need to combine ACCL with one or more FPGA compute kernels to perform useful work. To build the ACCL kernel by itself:
````
git clone https://github.com/Xilinx/ACCL.git
cd accl/kernels/cclo
<source Vivado 2020.2 settings script>
make BOARD=<u250|u280> 
cd ../../demo/build
<source Vivado 2020.1 settings script>
make PLATFORM=<shell name> tcp_stack_ips
````
This produces the CCLO Vitis kernel in `accl/kernels/cclo/ccl_offload.xo` and the TCP POE kernels in `demo/build/Vitis_with_100Gbps_TCP-IP/_x.hw.<shell name>/network_krnl.xo` and `/demo/build/Vitis_with_100Gbps_TCP-IP/_x.hw.$(XSA)/cmac_krnl.xo`. These three kernels can now be linked agains user kernels with Vitis. The following Vitis configuration file snippet instantiates the ACCL kernels and connects them appropriately:
````
# Instantiate ACCL kernels
nk=network_krnl:1:network_krnl_0
nk=ccl_offload:1:ccl_offload_0
nk=cmac_krnl:1:cmac_krnl_0

# Connect CMAC kernel to TCP Network kernel
sc=cmac_krnl_0.axis_net_rx:network_krnl_0.axis_net_rx
sc=network_krnl_0.axis_net_tx:cmac_krnl_0.axis_net_tx

# Connect CCL Offload kernel to TCP Network Kernel
sc=network_krnl_0.m_axis_tcp_port_status:ccl_offload_0.s_axis_tcp_port_status:512
sc=network_krnl_0.m_axis_tcp_open_status:ccl_offload_0.s_axis_tcp_open_status:512
sc=network_krnl_0.m_axis_tcp_notification:ccl_offload_0.s_axis_tcp_notification:512
sc=network_krnl_0.m_axis_tcp_rx_meta:ccl_offload_0.s_axis_tcp_rx_meta:512
sc=network_krnl_0.m_axis_tcp_rx_data:ccl_offload_0.s_axis_tcp_rx_data:512
sc=network_krnl_0.m_axis_tcp_tx_status:ccl_offload_0.s_axis_tcp_tx_status:512
sc=ccl_offload_0.m_axis_tcp_listen_port:network_krnl_0.s_axis_tcp_listen_port:512
sc=ccl_offload_0.m_axis_tcp_open_connection:network_krnl_0.s_axis_tcp_open_connection:512
sc=ccl_offload_0.m_axis_tcp_read_pkg:network_krnl_0.s_axis_tcp_read_pkg:512
sc=ccl_offload_0.m_axis_tcp_tx_meta:network_krnl_0.s_axis_tcp_tx_meta:512
sc=ccl_offload_0.m_axis_tcp_tx_data:network_krnl_0.s_axis_tcp_tx_data:512
````
We highly recommend explicit floorplanning memory bank assignment for the CCLO and POE, for timing closure and performance optimization. See configuration scripts in `accl/demo/build/config` for inspiration.