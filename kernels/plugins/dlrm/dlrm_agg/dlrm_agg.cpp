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

void dlrm_agg(
    int *dst,
    int count,
    // //parameters pertaining to CCLO config
    // ap_uint<32> comm_adr, 
    // ap_uint<32> dpcfg_adr,
    // //streams to and from CCLO
    // STREAM<command_word> &cmd_to_cclo,
    // STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
){
#pragma HLS INTERFACE s_axilite port=count
// #pragma HLS INTERFACE s_axilite port=comm_adr
// #pragma HLS INTERFACE s_axilite port=dpcfg_adr
#pragma HLS INTERFACE m_axi port=dst offset=slave
// #pragma HLS INTERFACE axis port=cmd_to_cclo
// #pragma HLS INTERFACE axis port=sts_from_cclo
#pragma HLS INTERFACE axis port=data_to_cclo
#pragma HLS INTERFACE axis port=data_from_cclo
#pragma HLS INTERFACE s_axilite port=return
    //set up interfaces
    // accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    //pull data from CCLO and write it to dst
    data.pull_to_mem(dst, count);
    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_agg: finish" << "\n";
    #endif
}