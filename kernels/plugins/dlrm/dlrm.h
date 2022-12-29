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

#include <accl_hls.h>
#include "constants.hpp"

#define DLRM_REDUCE_SLAVE_ROLE 0
#define DLRM_REDUCE_ROOT_ROLE 1
#define DLRM_EMBED_ROLE 2
#define DLRM_AGG_ROLE 3


void dlrm_agg(
    int *dst,
    // //parameters pertaining to CCLO config
    ap_uint<32> local_rank,
    ap_uint<32> comm_size,
    ap_uint<32> comm_adr, 
    ap_uint<32> dpcfg_adr,
    // //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
);

void dlrm_embedding(
    // dlrm parameters
    int *table_HBM0,
    int *table_HBM1,
    int *table_HBM2,
    int *table_HBM3,
    int *table_HBM4,
    int *table_HBM5,
    int *table_HBM6,
    int *table_HBM7,
    unsigned int destination,
    //parameters pertaining to CCLO config
    ap_uint<32> local_rank,
    ap_uint<32> comm_size,
    ap_uint<32> comm_adr, 
    ap_uint<32> dpcfg_adr,
    //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
);

void dlrm_reduce_root(
    unsigned int destination,
    //reduce configuration
    unsigned int root,
    unsigned int function,
    //parameters pertaining to CCLO config
    ap_uint<32> local_rank,
    ap_uint<32> global_comm_size,
    ap_uint<32> global_comm_adr, 
    ap_uint<32> reduce_comm_adr, 
    ap_uint<32> dpcfg_adr,
    //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
);

void dlrm_reduce_slave(
    //reduce configuration
    unsigned int root,
    unsigned int function,
    //parameters pertaining to CCLO config
    ap_uint<32> local_rank,
    ap_uint<32> global_comm_size,
    ap_uint<32> global_comm_adr, 
    ap_uint<32> reduce_comm_adr, 
    ap_uint<32> dpcfg_adr,
    //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
);

void dlrm_embedding_reduce(
    int *table_HBM0,
    int *table_HBM1,
    int *table_HBM2,
    int *table_HBM3,
    int *table_HBM4,
    int *table_HBM5,
    int *table_HBM6,
    int *table_HBM7,
    int *table_HBM8,
    int *table_HBM9,
    int *table_HBM10,
    int *table_HBM11,
    int *table_HBM12,
    int *table_HBM13,
    int *table_HBM14,
    int *table_HBM15,

    int count,
    unsigned int destination,
    //parameters pertaining to CCLO config
    ap_uint<32> comm_adr, 
    ap_uint<32> dpcfg_adr,
    //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
);
