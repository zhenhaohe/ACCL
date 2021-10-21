
# How to build it
The CCLO build process is automated and organized via Makefiles that are distributed across the repo. The build process is split into 2 steps:

1. building the CCLO IP. The main Makefile is [../kernel/Makefile](/kernel/Makefile). After this process an ``ccl_offload.xo`` file would be created under ``../kernel/ccl_offload_ex/exports``. 
For more info on how the Makefile works take a look in [../kernel/readme.md# Building and package the ccl_offload.xo Kernel](../kernel/readme.md#Building-and-package-the-ccl_offload.xo-Kernel).
2. Building the network stack and link against CCLO. The main Makefile is [../demo/build/Makefile](../demo/build/Makefile).

You can find info on how to build under [../demo/build](../demo/build).

Alveo shell currently supported:

- ``xilinx_u250_gen3x16_xdma_3_1_202020_1``
- ``xilinx_u280_xdma_201920_3``

# How to integrate it

Once you have built CCL_Offload and packaged into an IP, you can include it in 
your Vitis ``config.ini`` file in order to connect it to other kernels.
Here follows an example of config file.

````
[connectivity]
# Define number of kernels and their name
nk=network_krnl:1:network_krnl_0
nk=ccl_offload:1:ccl_offload_0
nk=cmac_krnl:1:cmac_krnl_0
nk=vnx_loopback:2:lb_str_0.lb_udp_0
nk=reduce_arith:1:external_reduce_arith_0

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

#Connect Network Kernel to CMAC Kernel
sc=cmac_krnl_0.axis_net_rx:network_krnl_0.axis_net_rx
sc=network_krnl_0.axis_net_tx:cmac_krnl_0.axis_net_tx
stream_connect=ccl_offload_0.m_axis_udp_tx_data:lb_udp_0.in
stream_connect=lb_udp_0.out:ccl_offload_0.s_axis_udp_rx_data

# Connect external reduce_arithmetic unit
stream_connect=external_reduce_arith_0.out_r:ccl_offload_0.s_axis_arith_res
stream_connect=ccl_offload_0.m_axis_arith_op0:external_reduce_arith_0.in1
stream_connect=ccl_offload_0.m_axis_arith_op1:external_reduce_arith_0.in2

# Connect external streaming kernel
stream_connect=ccl_offload_0.m_axis_krnl:lb_str_0.in
stream_connect=lb_str_0.out:ccl_offload_0.s_axis_krnl

````

An example on how to integrate the CCLO and the network stack is given at [../demo/Makefile](/demo/Makefile).

