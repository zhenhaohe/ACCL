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

void dlrm_reduce_slave(
    int count,
    //reduce configuration
    unsigned int root,
    unsigned int function,
    //parameters pertaining to CCLO config
    ap_uint<32> comm_adr, 
    ap_uint<32> dpcfg_adr,
    //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
){
#pragma HLS INTERFACE s_axilite port=count
#pragma HLS INTERFACE s_axilite port=root
#pragma HLS INTERFACE s_axilite port=function
#pragma HLS INTERFACE s_axilite port=comm_adr
#pragma HLS INTERFACE s_axilite port=dpcfg_adr
#pragma HLS INTERFACE axis port=cmd_to_cclo
#pragma HLS INTERFACE axis port=sts_from_cclo
#pragma HLS INTERFACE axis port=data_to_cclo
#pragma HLS INTERFACE axis port=data_from_cclo
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS dataflow disable_start_propagation

    //set up interfaces
    accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3); 
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    //streaming reduce command to CCLO
    // accl.reduce_nb(count, root, function);
    accl.reduce_put_nb(count, root, function);
    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_reduce_slave: reduce count=" << count << " root=" << root << "\n";
    #endif

    static STREAM<ap_uint<512> > stream_buf;
    #pragma HLS STREAM variable=stream_buf depth=512

    //pull data from CCLO and write it to stream_buf
    data.pull_to_stream(stream_buf, count);

    // compute 
    
    // read data from stream_buf and put to CCLO for reduction
    data.push_from_stream(stream_buf, count, 0);

    accl.finalize_call();
    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_reduce_slave: finish reduce" << "\n";
    #endif
}
