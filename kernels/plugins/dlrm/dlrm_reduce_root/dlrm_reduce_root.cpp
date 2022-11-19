/*******************************************************************************
#  Copyright (C) 2022 Advanced Micro Devices, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
*******************************************************************************/

#include <dlrm.h>

#ifdef DATA_FLOW

// void dlrm_reduce_root_command(
//     int *src,
//     int *dst,
//     int count,
//     unsigned int destination,
//     unsigned int root,
//     unsigned int function,
//     //parameters pertaining to CCLO config
//     ap_uint<32> comm_adr_global, 
//     ap_uint<32> comm_adr_reduce, 
//     ap_uint<32> dpcfg_adr,
//     //streams to and from CCLO
//     STREAM<command_word> &cmd_to_cclo,
//     STREAM<command_word> &sts_from_cclo
// )
// {
// #pragma HLS INLINE off
// #pragma HLS PIPELINE

//     //set up interfaces
//     accl_hls::ACCLCommand accl_reduce(cmd_to_cclo, sts_from_cclo, comm_adr_reduce, dpcfg_adr, 0, 2); // reduce communicator with streaming res interfaces
//     //set up interfaces
//     accl_hls::ACCLCommand accl_global(cmd_to_cclo, sts_from_cclo, comm_adr_global, dpcfg_adr, 0, 3); // global communicator with streaming interfaces

//     //reduce command to CCLO
//     accl_reduce.reduce_nb(count, root, function, (ap_uint<64>)src, dst); // dst address is not useful
//     #ifndef ACCL_SYNTHESIS
//         std::cout << "dlrm_reduce_root: reduce_nb command to CCLO" << "\n";
//     #endif
//     //send command to CCLO
//     //we're passing dst as source and targeting stream 9
//     //because we're streaming data, the address will be ignored
//     accl_global.stream_put_nb(count, 9, destination, (ap_uint<64>)dst); 
//     #ifndef ACCL_SYNTHESIS
//         std::cout << "dlrm_reduce_root: stream_put_nb command to CCLO" << "\n";
//     #endif

//     accl_reduce.finalize_call();
//     #ifndef ACCL_SYNTHESIS
//         std::cout << "dlrm_reduce_root: finish reduce" << "\n";
//     #endif
//     accl_global.finalize_call();
//     #ifndef ACCL_SYNTHESIS
//         std::cout << "dlrm_reduce_root: finish stream push" << "\n";
//     #endif
// }

void dlrm_reduce_root_cmd(
    ap_uint<64> src_addr,
    int count,
    unsigned int destination,
    unsigned int root,
    unsigned int function,
    //parameters pertaining to CCLO config
    ap_uint<32> comm_adr_global, 
    ap_uint<32> comm_adr_reduce, 
    ap_uint<32> dpcfg_adr,
    //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo
)
{
#pragma HLS INLINE off
#pragma HLS PIPELINE

    accl_hls::start(ACCL_REDUCE, count, comm_adr_reduce, root, function, 0, dpcfg_adr, 0, 2, (ap_uint<64>)src_addr, 0, 0, cmd_to_cclo);
    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_reduce_root: reduce_nb command to CCLO" << "\n";
    #endif
    accl_hls::start(ACCL_SEND, count, comm_adr_global, destination, function, 9, dpcfg_adr, 0, 3, 0, 0, 0, cmd_to_cclo);
    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_reduce_root: stream_put_nb command to CCLO" << "\n";
    #endif
}


void dlrm_reduce_root_ack(STREAM<command_word> &sts_from_cclo)
{
#pragma HLS INLINE off
#pragma HLS PIPELINE

    accl_hls::finalize(sts_from_cclo);
    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_reduce_root: finish reduce" << "\n";
    #endif
    accl_hls::finalize(sts_from_cclo);
    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_reduce_root: finish stream push" << "\n";
    #endif
}

void dlrm_reduce_root(
    ap_uint<64> src_addr,
    int count,
    unsigned int destination,
    //reduce configuration
    unsigned int root,
    unsigned int function,
    //parameters pertaining to CCLO config
    ap_uint<32> comm_adr_global, 
    ap_uint<32> comm_adr_reduce, 
    ap_uint<32> dpcfg_adr,
    //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
){
#pragma HLS INTERFACE s_axilite port=src_addr
#pragma HLS INTERFACE s_axilite port=count
#pragma HLS INTERFACE s_axilite port=destination
#pragma HLS INTERFACE s_axilite port=root
#pragma HLS INTERFACE s_axilite port=function
#pragma HLS INTERFACE s_axilite port=comm_adr_global
#pragma HLS INTERFACE s_axilite port=comm_adr_reduce
#pragma HLS INTERFACE s_axilite port=dpcfg_adr
#pragma HLS INTERFACE axis port=cmd_to_cclo
#pragma HLS INTERFACE axis port=sts_from_cclo
#pragma HLS INTERFACE axis port=data_to_cclo
#pragma HLS INTERFACE axis port=data_from_cclo
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS dataflow disable_start_propagation

    dlrm_reduce_root_cmd( src_addr, count, destination, root, function, comm_adr_global, comm_adr_reduce, dpcfg_adr, cmd_to_cclo);

    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    
    static STREAM<ap_uint<512> > stream_buf;
    #pragma HLS STREAM variable=stream_buf depth=512

    //pull data from CCLO and write it to stream_buf
    data.pull_to_stream(stream_buf, count);
    // read data from stream_buf and put to CCLO for reduction
    data.push_from_stream(stream_buf, count, 0);

    dlrm_reduce_root_ack(sts_from_cclo);
}

#else

void dlrm_reduce_root(
    int *src,
    int *dst,
    int count,
    unsigned int destination,
    //reduce configuration
    unsigned int root,
    unsigned int function,
    //parameters pertaining to CCLO config
    ap_uint<32> comm_adr_global, 
    ap_uint<32> comm_adr_reduce, 
    ap_uint<32> dpcfg_adr,
    //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
){
#pragma HLS INTERFACE s_axilite port=count
#pragma HLS INTERFACE s_axilite port=destination
#pragma HLS INTERFACE s_axilite port=root
#pragma HLS INTERFACE s_axilite port=function
#pragma HLS INTERFACE s_axilite port=comm_adr_global
#pragma HLS INTERFACE s_axilite port=comm_adr_reduce
#pragma HLS INTERFACE s_axilite port=dpcfg_adr
#pragma HLS INTERFACE m_axi port=src offset=slave
#pragma HLS INTERFACE m_axi port=dst offset=slave
#pragma HLS INTERFACE axis port=cmd_to_cclo
#pragma HLS INTERFACE axis port=sts_from_cclo
#pragma HLS INTERFACE axis port=data_to_cclo
#pragma HLS INTERFACE axis port=data_from_cclo
#pragma HLS INTERFACE s_axilite port=return
    //set up interfaces
    accl_hls::ACCLCommand accl_global(cmd_to_cclo, sts_from_cclo, comm_adr_global, dpcfg_adr, 0, 3); // global communicator with streaming interfaces
    accl_hls::ACCLCommand accl_reduce(cmd_to_cclo, sts_from_cclo, comm_adr_reduce, dpcfg_adr, 0, 3); // reduce communicator with non-streaming interfaces
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    //reduce command to CCLO
    accl_reduce.reduce_nb(count, root, function); 
    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_reduce_root: reduce command to CCLO" << "\n";
    #endif
    // read src and send it to CCLO as stream
    data.push_from_mem(src, count, 0);
    // pull the stream from the CCLO and store it to dst
    data.pull_to_mem(dst, count);
    // reduce finished with data store in dst
    accl_reduce.finalize_call();
    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_reduce_root: finish reduce" << "\n";
    #endif
    //send command to CCLO
    //we're passing dst as source and targeting stream 9
    //because we're streaming data, the address will be ignored
    accl_global.stream_put_nb(count, 9, destination, (ap_uint<64>)dst); 
    //read data from dst
    //and push the result into the CCLO stream
    data.push_from_mem(dst, count, 0);
    accl_global.finalize_call();
    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_reduce_root: finish stream push" << "\n";
    #endif
}

#endif