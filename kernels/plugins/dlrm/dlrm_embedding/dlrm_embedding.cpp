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
#include "compute_embedding.hpp"

using namespace dlrm_embedding_ns;

void dlrm_embedding_compute(
    int *table_HBM0,
    int *table_HBM1,
    int *table_HBM2,
    int *table_HBM3,
    int *table_HBM4,
    int *table_HBM5,
    int *table_HBM6,
    int *table_HBM7,
    unsigned int destination,
    ap_uint<32> comm_adr, 
    ap_uint<32> dpcfg_adr,
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo)
{
    STREAM<t_axi> s_embedding_buffer_HBM0;
    STREAM<t_axi> s_embedding_buffer_HBM1;
    STREAM<t_axi> s_embedding_buffer_HBM2;
    STREAM<t_axi> s_embedding_buffer_HBM3;
    STREAM<t_axi> s_embedding_buffer_HBM4;
    STREAM<t_axi> s_embedding_buffer_HBM5;
    STREAM<t_axi> s_embedding_buffer_HBM6;
    STREAM<t_axi> s_embedding_buffer_HBM7;

#pragma HLS stream variable=s_embedding_buffer_HBM0 depth=depth_s_embedding_buffer_HBM0
#pragma HLS stream variable=s_embedding_buffer_HBM1 depth=depth_s_embedding_buffer_HBM1
#pragma HLS stream variable=s_embedding_buffer_HBM2 depth=depth_s_embedding_buffer_HBM2
#pragma HLS stream variable=s_embedding_buffer_HBM3 depth=depth_s_embedding_buffer_HBM3
#pragma HLS stream variable=s_embedding_buffer_HBM4 depth=depth_s_embedding_buffer_HBM4
#pragma HLS stream variable=s_embedding_buffer_HBM5 depth=depth_s_embedding_buffer_HBM5
#pragma HLS stream variable=s_embedding_buffer_HBM6 depth=depth_s_embedding_buffer_HBM6
#pragma HLS stream variable=s_embedding_buffer_HBM7 depth=depth_s_embedding_buffer_HBM7

    STREAM<W_TYPE> s_embedding_buffer_wide_HBM0_1;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM1_1;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM2_1;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM3_1;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM4_1;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM5_1;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM6_1;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM7_1;

#pragma HLS stream variable=s_embedding_buffer_wide_HBM0_1 depth=depth_s_embedding_buffer_wide_HBM0
#pragma HLS stream variable=s_embedding_buffer_wide_HBM1_1 depth=depth_s_embedding_buffer_wide_HBM1
#pragma HLS stream variable=s_embedding_buffer_wide_HBM2_1 depth=depth_s_embedding_buffer_wide_HBM2
#pragma HLS stream variable=s_embedding_buffer_wide_HBM3_1 depth=depth_s_embedding_buffer_wide_HBM3
#pragma HLS stream variable=s_embedding_buffer_wide_HBM4_1 depth=depth_s_embedding_buffer_wide_HBM4
#pragma HLS stream variable=s_embedding_buffer_wide_HBM5_1 depth=depth_s_embedding_buffer_wide_HBM5
#pragma HLS stream variable=s_embedding_buffer_wide_HBM6_1 depth=depth_s_embedding_buffer_wide_HBM6
#pragma HLS stream variable=s_embedding_buffer_wide_HBM7_1 depth=depth_s_embedding_buffer_wide_HBM7

    STREAM<W_TYPE> s_embedding_buffer_wide_HBM0_2;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM1_2;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM2_2;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM3_2;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM4_2;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM5_2;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM6_2;
    STREAM<W_TYPE> s_embedding_buffer_wide_HBM7_2;

#pragma HLS stream variable=s_embedding_buffer_wide_HBM0_2 depth=depth_s_embedding_buffer_wide_HBM0
#pragma HLS stream variable=s_embedding_buffer_wide_HBM1_2 depth=depth_s_embedding_buffer_wide_HBM1
#pragma HLS stream variable=s_embedding_buffer_wide_HBM2_2 depth=depth_s_embedding_buffer_wide_HBM2
#pragma HLS stream variable=s_embedding_buffer_wide_HBM3_2 depth=depth_s_embedding_buffer_wide_HBM3
#pragma HLS stream variable=s_embedding_buffer_wide_HBM4_2 depth=depth_s_embedding_buffer_wide_HBM4
#pragma HLS stream variable=s_embedding_buffer_wide_HBM5_2 depth=depth_s_embedding_buffer_wide_HBM5
#pragma HLS stream variable=s_embedding_buffer_wide_HBM6_2 depth=depth_s_embedding_buffer_wide_HBM6
#pragma HLS stream variable=s_embedding_buffer_wide_HBM7_2 depth=depth_s_embedding_buffer_wide_HBM7

    STREAM<int> s_idx_buffer_HBM0;
    STREAM<int> s_idx_buffer_HBM1;
    STREAM<int> s_idx_buffer_HBM2;
    STREAM<int> s_idx_buffer_HBM3;
    STREAM<int> s_idx_buffer_HBM4;
    STREAM<int> s_idx_buffer_HBM5;
    STREAM<int> s_idx_buffer_HBM6;
    STREAM<int> s_idx_buffer_HBM7;

#pragma HLS stream variable=s_idx_buffer_HBM0 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM1 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM2 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM3 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM4 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM5 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM6 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM7 depth=fifo_batch_size

    STREAM<ap_uint<512> > s_embedding_0;
#pragma HLS stream variable=s_embedding_0 depth=256

    STREAM<ap_uint<512> > s_feature_in;
#pragma HLS stream variable=s_feature_in depth=512
    STREAM<ap_uint<512> > s_feature_out;
#pragma HLS stream variable=s_feature_out depth=512

    STREAM<W_TYPE> s_feature1_PE0_0;
    STREAM<W_TYPE> s_feature1_PE0_1;
    STREAM<D_TYPE> s_result1_PE0;
#pragma HLS stream variable=s_feature1_PE0_0 depth=2
#pragma HLS stream variable=s_feature1_PE0_1 depth=2
#pragma HLS stream variable=s_result1_PE0 depth=2
    STREAM<W_TYPE> s_feature1_PE1_0;
    STREAM<W_TYPE> s_feature1_PE1_1;
    STREAM<D_TYPE> s_result1_PE1;
#pragma HLS stream variable=s_feature1_PE1_0 depth=2
#pragma HLS stream variable=s_feature1_PE1_1 depth=2
#pragma HLS stream variable=s_result1_PE1 depth=2
    STREAM<W_TYPE> s_feature1_PE2_0;
    STREAM<W_TYPE> s_feature1_PE2_1;
    STREAM<D_TYPE> s_result1_PE2;
#pragma HLS stream variable=s_feature1_PE2_0 depth=2
#pragma HLS stream variable=s_feature1_PE2_1 depth=2
#pragma HLS stream variable=s_result1_PE2 depth=2
    STREAM<W_TYPE> s_feature1_PE3_0;
    STREAM<W_TYPE> s_feature1_PE3_1;
    STREAM<D_TYPE> s_result1_PE3;
#pragma HLS stream variable=s_feature1_PE3_0 depth=2
#pragma HLS stream variable=s_feature1_PE3_1 depth=2
#pragma HLS stream variable=s_result1_PE3 depth=2
    STREAM<W_TYPE> s_feature1_PE4_0;
    STREAM<W_TYPE> s_feature1_PE4_1;
    STREAM<D_TYPE> s_result1_PE4;
#pragma HLS stream variable=s_feature1_PE4_0 depth=2
#pragma HLS stream variable=s_feature1_PE4_1 depth=2
#pragma HLS stream variable=s_result1_PE4 depth=2
    STREAM<W_TYPE> s_feature1_PE5_0;
    STREAM<W_TYPE> s_feature1_PE5_1;
    STREAM<D_TYPE> s_result1_PE5;
#pragma HLS stream variable=s_feature1_PE5_0 depth=2
#pragma HLS stream variable=s_feature1_PE5_1 depth=2
#pragma HLS stream variable=s_result1_PE5 depth=2
    STREAM<W_TYPE> s_feature1_PE6_0;
    STREAM<W_TYPE> s_feature1_PE6_1;
    STREAM<D_TYPE> s_result1_PE6;
#pragma HLS stream variable=s_feature1_PE6_0 depth=2
#pragma HLS stream variable=s_feature1_PE6_1 depth=2
#pragma HLS stream variable=s_result1_PE6 depth=2
    STREAM<W_TYPE> s_feature1_PE7_0;
    STREAM<W_TYPE> s_feature1_PE7_1;
    STREAM<D_TYPE> s_result1_PE7;
#pragma HLS stream variable=s_feature1_PE7_0 depth=2
#pragma HLS stream variable=s_feature1_PE7_1 depth=2
#pragma HLS stream variable=s_result1_PE7 depth=2
    STREAM<W_TYPE> s_feature1_PE8_0;
    STREAM<W_TYPE> s_feature1_PE8_1;
    STREAM<D_TYPE> s_result1_PE8;
#pragma HLS stream variable=s_feature1_PE8_0 depth=2
#pragma HLS stream variable=s_feature1_PE8_1 depth=2
#pragma HLS stream variable=s_result1_PE8 depth=2
    STREAM<W_TYPE> s_feature1_PE9_0;
    STREAM<W_TYPE> s_feature1_PE9_1;
    STREAM<D_TYPE> s_result1_PE9;
#pragma HLS stream variable=s_feature1_PE9_0 depth=2
#pragma HLS stream variable=s_feature1_PE9_1 depth=2
#pragma HLS stream variable=s_result1_PE9 depth=2
    STREAM<W_TYPE> s_feature1_PE10_0;
    STREAM<W_TYPE> s_feature1_PE10_1;
    STREAM<D_TYPE> s_result1_PE10;
#pragma HLS stream variable=s_feature1_PE10_0 depth=2
#pragma HLS stream variable=s_feature1_PE10_1 depth=2
#pragma HLS stream variable=s_result1_PE10 depth=2
    STREAM<W_TYPE> s_feature1_PE11_0;
    STREAM<W_TYPE> s_feature1_PE11_1;
    STREAM<D_TYPE> s_result1_PE11;
#pragma HLS stream variable=s_feature1_PE11_0 depth=2
#pragma HLS stream variable=s_feature1_PE11_1 depth=2
#pragma HLS stream variable=s_result1_PE11 depth=2
    STREAM<W_TYPE> s_feature1_PE12_0;
    STREAM<W_TYPE> s_feature1_PE12_1;
    STREAM<D_TYPE> s_result1_PE12;
#pragma HLS stream variable=s_feature1_PE12_0 depth=2
#pragma HLS stream variable=s_feature1_PE12_1 depth=2
#pragma HLS stream variable=s_result1_PE12 depth=2
    STREAM<W_TYPE> s_feature1_PE13_0;
    STREAM<W_TYPE> s_feature1_PE13_1;
    STREAM<D_TYPE> s_result1_PE13;
#pragma HLS stream variable=s_feature1_PE13_0 depth=2
#pragma HLS stream variable=s_feature1_PE13_1 depth=2
#pragma HLS stream variable=s_result1_PE13 depth=2
    STREAM<W_TYPE> s_feature1_PE14_0;
    STREAM<W_TYPE> s_feature1_PE14_1;
    STREAM<D_TYPE> s_result1_PE14;
#pragma HLS stream variable=s_feature1_PE14_0 depth=2
#pragma HLS stream variable=s_feature1_PE14_1 depth=2
#pragma HLS stream variable=s_result1_PE14 depth=2
    STREAM<W_TYPE> s_feature1_PE15_0;
    STREAM<W_TYPE> s_feature1_PE15_1;
    STREAM<D_TYPE> s_result1_PE15;
#pragma HLS stream variable=s_feature1_PE15_0 depth=2
#pragma HLS stream variable=s_feature1_PE15_1 depth=2
#pragma HLS stream variable=s_result1_PE15 depth=2
    STREAM<W_TYPE> s_feature1_PE16_0;
    STREAM<W_TYPE> s_feature1_PE16_1;
    STREAM<D_TYPE> s_result1_PE16;
#pragma HLS stream variable=s_feature1_PE16_0 depth=2
#pragma HLS stream variable=s_feature1_PE16_1 depth=2
#pragma HLS stream variable=s_result1_PE16 depth=2
    STREAM<W_TYPE> s_feature1_PE17_0;
    STREAM<W_TYPE> s_feature1_PE17_1;
    STREAM<D_TYPE> s_result1_PE17;
#pragma HLS stream variable=s_feature1_PE17_0 depth=2
#pragma HLS stream variable=s_feature1_PE17_1 depth=2
#pragma HLS stream variable=s_result1_PE17 depth=2
    STREAM<W_TYPE> s_feature1_PE18_0;
    STREAM<W_TYPE> s_feature1_PE18_1;
    STREAM<D_TYPE> s_result1_PE18;
#pragma HLS stream variable=s_feature1_PE18_0 depth=2
#pragma HLS stream variable=s_feature1_PE18_1 depth=2
#pragma HLS stream variable=s_result1_PE18 depth=2
    STREAM<W_TYPE> s_feature1_PE19_0;
    STREAM<W_TYPE> s_feature1_PE19_1;
    STREAM<D_TYPE> s_result1_PE19;
#pragma HLS stream variable=s_feature1_PE19_0 depth=2
#pragma HLS stream variable=s_feature1_PE19_1 depth=2
#pragma HLS stream variable=s_result1_PE19 depth=2
    STREAM<W_TYPE> s_feature1_PE20_0;
    STREAM<W_TYPE> s_feature1_PE20_1;
    STREAM<D_TYPE> s_result1_PE20;
#pragma HLS stream variable=s_feature1_PE20_0 depth=2
#pragma HLS stream variable=s_feature1_PE20_1 depth=2
#pragma HLS stream variable=s_result1_PE20 depth=2
    STREAM<W_TYPE> s_feature1_PE21_0;
    STREAM<W_TYPE> s_feature1_PE21_1;
    STREAM<D_TYPE> s_result1_PE21;
#pragma HLS stream variable=s_feature1_PE21_0 depth=2
#pragma HLS stream variable=s_feature1_PE21_1 depth=2
#pragma HLS stream variable=s_result1_PE21 depth=2
    STREAM<W_TYPE> s_feature1_PE22_0;
    STREAM<W_TYPE> s_feature1_PE22_1;
    STREAM<D_TYPE> s_result1_PE22;
#pragma HLS stream variable=s_feature1_PE22_0 depth=2
#pragma HLS stream variable=s_feature1_PE22_1 depth=2
#pragma HLS stream variable=s_result1_PE22 depth=2
    STREAM<W_TYPE> s_feature1_PE23_0;
    STREAM<W_TYPE> s_feature1_PE23_1;
    STREAM<D_TYPE> s_result1_PE23;
#pragma HLS stream variable=s_feature1_PE23_0 depth=2
#pragma HLS stream variable=s_feature1_PE23_1 depth=2
#pragma HLS stream variable=s_result1_PE23 depth=2
    STREAM<W_TYPE> s_feature1_PE24_0;
    STREAM<W_TYPE> s_feature1_PE24_1;
    STREAM<D_TYPE> s_result1_PE24;
#pragma HLS stream variable=s_feature1_PE24_0 depth=2
#pragma HLS stream variable=s_feature1_PE24_1 depth=2
#pragma HLS stream variable=s_result1_PE24 depth=2
    STREAM<W_TYPE> s_feature1_PE25_0;
    STREAM<W_TYPE> s_feature1_PE25_1;
    STREAM<D_TYPE> s_result1_PE25;
#pragma HLS stream variable=s_feature1_PE25_0 depth=2
#pragma HLS stream variable=s_feature1_PE25_1 depth=2
#pragma HLS stream variable=s_result1_PE25 depth=2
    STREAM<W_TYPE> s_feature1_PE26_0;
    STREAM<W_TYPE> s_feature1_PE26_1;
    STREAM<D_TYPE> s_result1_PE26;
#pragma HLS stream variable=s_feature1_PE26_0 depth=2
#pragma HLS stream variable=s_feature1_PE26_1 depth=2
#pragma HLS stream variable=s_result1_PE26 depth=2
    STREAM<W_TYPE> s_feature1_PE27_0;
    STREAM<W_TYPE> s_feature1_PE27_1;
    STREAM<D_TYPE> s_result1_PE27;
#pragma HLS stream variable=s_feature1_PE27_0 depth=2
#pragma HLS stream variable=s_feature1_PE27_1 depth=2
#pragma HLS stream variable=s_result1_PE27 depth=2
    STREAM<W_TYPE> s_feature1_PE28_0;
    STREAM<W_TYPE> s_feature1_PE28_1;
    STREAM<D_TYPE> s_result1_PE28;
#pragma HLS stream variable=s_feature1_PE28_0 depth=2
#pragma HLS stream variable=s_feature1_PE28_1 depth=2
#pragma HLS stream variable=s_result1_PE28 depth=2
    STREAM<W_TYPE> s_feature1_PE29_0;
    STREAM<W_TYPE> s_feature1_PE29_1;
    STREAM<D_TYPE> s_result1_PE29;
#pragma HLS stream variable=s_feature1_PE29_0 depth=2
#pragma HLS stream variable=s_feature1_PE29_1 depth=2
#pragma HLS stream variable=s_result1_PE29 depth=2
    STREAM<W_TYPE> s_feature1_PE30_0;
    STREAM<W_TYPE> s_feature1_PE30_1;
    STREAM<D_TYPE> s_result1_PE30;
#pragma HLS stream variable=s_feature1_PE30_0 depth=2
#pragma HLS stream variable=s_feature1_PE30_1 depth=2
#pragma HLS stream variable=s_result1_PE30 depth=2
    STREAM<W_TYPE> s_feature1_PE31_0;
    STREAM<W_TYPE> s_feature1_PE31_1;
    STREAM<D_TYPE> s_result1_PE31;

    STREAM<ap_uint<512> > s_result1_partial_0;
#pragma HLS stream variable=s_result1_partial_0 depth=128

    STREAM<ap_uint<512> > s_result1_node1;
#pragma HLS stream variable=s_result1_node1 depth=256

    STREAM<ap_uint<512> >    s_embedding_table;
#pragma HLS STREAM variable=s_embedding_table depth=512

    STREAM<ap_uint<512> >    s_padded_zero;
#pragma HLS STREAM variable=s_padded_zero depth=512

    STREAM<ap_uint<512> >    s_data_out;
#pragma HLS STREAM variable=s_data_out depth=512

#pragma HLS dataflow disable_start_propagation

    load_access_idx(
        s_idx_buffer_HBM0, s_idx_buffer_HBM1, s_idx_buffer_HBM2, s_idx_buffer_HBM3, 
        s_idx_buffer_HBM4, s_idx_buffer_HBM5, s_idx_buffer_HBM6, s_idx_buffer_HBM7
        );

    load_single_embedding_2_tables<ADDR_AXI_HBM_0, AXI_PADDED_SIZE_HBM_0, ADDR_AXI_HBM_32, AXI_PADDED_SIZE_HBM_32, EMBEDDING_ROW_PER_PE1>(
        s_idx_buffer_HBM0, table_HBM0, s_embedding_buffer_HBM0);
    load_single_embedding_2_tables<ADDR_AXI_HBM_1, AXI_PADDED_SIZE_HBM_1, ADDR_AXI_HBM_33, AXI_PADDED_SIZE_HBM_33, EMBEDDING_ROW_PER_PE1>(
        s_idx_buffer_HBM1, table_HBM1, s_embedding_buffer_HBM1);
    load_single_embedding_2_tables<ADDR_AXI_HBM_2, AXI_PADDED_SIZE_HBM_2, ADDR_AXI_HBM_34, AXI_PADDED_SIZE_HBM_34, EMBEDDING_ROW_PER_PE1>(
        s_idx_buffer_HBM2, table_HBM2, s_embedding_buffer_HBM2);
    load_single_embedding_2_tables<ADDR_AXI_HBM_3, AXI_PADDED_SIZE_HBM_3, ADDR_AXI_HBM_35, AXI_PADDED_SIZE_HBM_35, EMBEDDING_ROW_PER_PE1>(
        s_idx_buffer_HBM3, table_HBM3, s_embedding_buffer_HBM3);
    load_single_embedding_2_tables<ADDR_AXI_HBM_4, AXI_PADDED_SIZE_HBM_4, ADDR_AXI_HBM_36, AXI_PADDED_SIZE_HBM_36, EMBEDDING_ROW_PER_PE1>(
        s_idx_buffer_HBM4, table_HBM4, s_embedding_buffer_HBM4);
    load_single_embedding_2_tables<ADDR_AXI_HBM_5, AXI_PADDED_SIZE_HBM_5, ADDR_AXI_HBM_37, AXI_PADDED_SIZE_HBM_37, EMBEDDING_ROW_PER_PE1>(
        s_idx_buffer_HBM5, table_HBM5, s_embedding_buffer_HBM5);
    load_single_embedding_2_tables<ADDR_AXI_HBM_6, AXI_PADDED_SIZE_HBM_6, ADDR_AXI_HBM_38, AXI_PADDED_SIZE_HBM_38, EMBEDDING_ROW_PER_PE1>(
        s_idx_buffer_HBM6, table_HBM6, s_embedding_buffer_HBM6);
    load_single_embedding_2_tables<ADDR_AXI_HBM_7, AXI_PADDED_SIZE_HBM_7, ADDR_AXI_HBM_39, AXI_PADDED_SIZE_HBM_39, EMBEDDING_ROW_PER_PE1>(
        s_idx_buffer_HBM7, table_HBM7, s_embedding_buffer_HBM7);

    int_to_wide<t_axi, VECTOR_SIZE_HBM_BANK_0, EMBEDDING_ROW_PER_PE1>(s_embedding_buffer_HBM0, s_embedding_buffer_wide_HBM0_1);
    int_to_wide<t_axi, VECTOR_SIZE_HBM_BANK_1, EMBEDDING_ROW_PER_PE1>(s_embedding_buffer_HBM1, s_embedding_buffer_wide_HBM1_1);
    int_to_wide<t_axi, VECTOR_SIZE_HBM_BANK_2, EMBEDDING_ROW_PER_PE1>(s_embedding_buffer_HBM2, s_embedding_buffer_wide_HBM2_1);
    int_to_wide<t_axi, VECTOR_SIZE_HBM_BANK_3, EMBEDDING_ROW_PER_PE1>(s_embedding_buffer_HBM3, s_embedding_buffer_wide_HBM3_1);
    int_to_wide<t_axi, VECTOR_SIZE_HBM_BANK_4, EMBEDDING_ROW_PER_PE1>(s_embedding_buffer_HBM4, s_embedding_buffer_wide_HBM4_1);
    int_to_wide<t_axi, VECTOR_SIZE_HBM_BANK_5, EMBEDDING_ROW_PER_PE1>(s_embedding_buffer_HBM5, s_embedding_buffer_wide_HBM5_1);
    int_to_wide<t_axi, VECTOR_SIZE_HBM_BANK_6, EMBEDDING_ROW_PER_PE1>(s_embedding_buffer_HBM6, s_embedding_buffer_wide_HBM6_1);
    int_to_wide<t_axi, VECTOR_SIZE_HBM_BANK_7, EMBEDDING_ROW_PER_PE1>(s_embedding_buffer_HBM7, s_embedding_buffer_wide_HBM7_1);

    gather_embeddings_8<VECTOR_SIZE_HBM_BANK_0, VECTOR_SIZE_HBM_BANK_1, VECTOR_SIZE_HBM_BANK_2, VECTOR_SIZE_HBM_BANK_3, VECTOR_SIZE_HBM_BANK_4, VECTOR_SIZE_HBM_BANK_5, VECTOR_SIZE_HBM_BANK_6, VECTOR_SIZE_HBM_BANK_7, EMBEDDING_ROW_PER_PE1>(
        s_embedding_buffer_wide_HBM0_1, 
        s_embedding_buffer_wide_HBM1_1, 
        s_embedding_buffer_wide_HBM2_1, 
        s_embedding_buffer_wide_HBM3_1, 
        s_embedding_buffer_wide_HBM4_1, 
        s_embedding_buffer_wide_HBM5_1, 
        s_embedding_buffer_wide_HBM6_1, 
        s_embedding_buffer_wide_HBM7_1,
        s_embedding_0);

    gather_embeddings<EMBEDDING_ROW_PER_PE1>(s_embedding_0, s_feature_in);

    // feature words is BATCH_NUM * BATCH_SIZE * FEATURE_SIZE / INTS_PER_W / 4 = 50
    store_features<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1>(s_feature_in, s_feature_out, s_embedding_table);

    replicate_feature_512PEs_32PE<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1>(
        s_feature_out, 
        s_feature1_PE0_0, s_feature1_PE0_1, s_feature1_PE1_0, s_feature1_PE1_1,
        s_feature1_PE2_0, s_feature1_PE2_1, s_feature1_PE3_0, s_feature1_PE3_1,
        s_feature1_PE4_0, s_feature1_PE4_1, s_feature1_PE5_0, s_feature1_PE5_1,
        s_feature1_PE6_0, s_feature1_PE6_1, s_feature1_PE7_0, s_feature1_PE7_1,
        s_feature1_PE8_0, s_feature1_PE8_1, s_feature1_PE9_0, s_feature1_PE9_1,
        s_feature1_PE10_0, s_feature1_PE10_1, s_feature1_PE11_0, s_feature1_PE11_1,
        s_feature1_PE12_0, s_feature1_PE12_1, s_feature1_PE13_0, s_feature1_PE13_1,
        s_feature1_PE14_0, s_feature1_PE14_1, s_feature1_PE15_0, s_feature1_PE15_1,
        s_feature1_PE16_0, s_feature1_PE16_1, s_feature1_PE17_0, s_feature1_PE17_1,
        s_feature1_PE18_0, s_feature1_PE18_1, s_feature1_PE19_0, s_feature1_PE19_1,
        s_feature1_PE20_0, s_feature1_PE20_1, s_feature1_PE21_0, s_feature1_PE21_1,
        s_feature1_PE22_0, s_feature1_PE22_1, s_feature1_PE23_0, s_feature1_PE23_1,
        s_feature1_PE24_0, s_feature1_PE24_1, s_feature1_PE25_0, s_feature1_PE25_1,
        s_feature1_PE26_0, s_feature1_PE26_1, s_feature1_PE27_0, s_feature1_PE27_1,
        s_feature1_PE28_0, s_feature1_PE28_1, s_feature1_PE29_0, s_feature1_PE29_1,
        s_feature1_PE30_0, s_feature1_PE30_1, s_feature1_PE31_0, s_feature1_PE31_1);

    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE0_0, s_feature1_PE0_1, s_result1_PE0);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE1_0, s_feature1_PE1_1, s_result1_PE1);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE2_0, s_feature1_PE2_1, s_result1_PE2);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE3_0, s_feature1_PE3_1, s_result1_PE3);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE4_0, s_feature1_PE4_1, s_result1_PE4);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE5_0, s_feature1_PE5_1, s_result1_PE5);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE6_0, s_feature1_PE6_1, s_result1_PE6);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE7_0, s_feature1_PE7_1, s_result1_PE7);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE8_0, s_feature1_PE8_1, s_result1_PE8);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE9_0, s_feature1_PE9_1, s_result1_PE9);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE10_0, s_feature1_PE10_1, s_result1_PE10);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE11_0, s_feature1_PE11_1, s_result1_PE11);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE12_0, s_feature1_PE12_1, s_result1_PE12);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE13_0, s_feature1_PE13_1, s_result1_PE13);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE14_0, s_feature1_PE14_1, s_result1_PE14);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE15_0, s_feature1_PE15_1, s_result1_PE15);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE16_0, s_feature1_PE16_1, s_result1_PE16);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE17_0, s_feature1_PE17_1, s_result1_PE17);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE18_0, s_feature1_PE18_1, s_result1_PE18);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE19_0, s_feature1_PE19_1, s_result1_PE19);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE20_0, s_feature1_PE20_1, s_result1_PE20);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE21_0, s_feature1_PE21_1, s_result1_PE21);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE22_0, s_feature1_PE22_1, s_result1_PE22);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE23_0, s_feature1_PE23_1, s_result1_PE23);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE24_0, s_feature1_PE24_1, s_result1_PE24);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE25_0, s_feature1_PE25_1, s_result1_PE25);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE26_0, s_feature1_PE26_1, s_result1_PE26);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE27_0, s_feature1_PE27_1, s_result1_PE27);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE28_0, s_feature1_PE28_1, s_result1_PE28);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE29_0, s_feature1_PE29_1, s_result1_PE29);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_BRAM>(s_feature1_PE30_0, s_feature1_PE30_1, s_result1_PE30);
    matmul_PE_UNROLL8<EMBEDDING_INPUT_SIZE, EMBEDDING_ROW_PER_PE1, WEIGHT_URAM>(s_feature1_PE31_0, s_feature1_PE31_1, s_result1_PE31);

    gather_results_32PEs<EMBEDDING_ROW_PER_PE1>(
        s_result1_PE0, s_result1_PE1, s_result1_PE2, s_result1_PE3,
        s_result1_PE4, s_result1_PE5, s_result1_PE6, s_result1_PE7,
        s_result1_PE8, s_result1_PE9, s_result1_PE10, s_result1_PE11,
        s_result1_PE12, s_result1_PE13, s_result1_PE14, s_result1_PE15,
        s_result1_PE16, s_result1_PE17, s_result1_PE18, s_result1_PE19,
        s_result1_PE20, s_result1_PE21, s_result1_PE22, s_result1_PE23,
        s_result1_PE24, s_result1_PE25, s_result1_PE26, s_result1_PE27,
        s_result1_PE28, s_result1_PE29, s_result1_PE30, s_result1_PE31,
        s_result1_partial_0);

    // creates 64 words 
    gather_results_node1<EMBEDDING_ROW_PER_PE1>(
        s_result1_partial_0,
        s_result1_node1);

    //padding 78 words
    pad_zero<78>(s_padded_zero);

    //embedding 50 word, results 64 words, data out is padded to 192 words
    dataTransform(s_embedding_table, s_result1_node1, s_padded_zero, s_data_out);

    int dlrm_count = BATCH_NUM * BATCH_SIZE * 3 * 64 * 16;

    //set up interfaces
    accl_hls::ACCLData data_dlrm(data_to_cclo, data_from_cclo);

    //send command to CCLO
    accl_hls::start(ACCL_SEND, dlrm_count, comm_adr, destination, 0, 9, dpcfg_adr, 0, 3, 0, 0, 0, cmd_to_cclo);

    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_embedding: stream_put count=" << dlrm_count << " destination=" << destination << "\n";
    #endif

    //and push the result into the CCLO stream
    data_dlrm.push_from_stream(s_data_out, dlrm_count, 0);
    // finalize the call
    accl_hls::finalize(sts_from_cclo);

    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_embedding: finish" << "\n";
    #endif

}

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
){
#pragma HLS INTERFACE s_axilite port=destination
#pragma HLS INTERFACE s_axilite port=local_rank
#pragma HLS INTERFACE s_axilite port=comm_size
#pragma HLS INTERFACE s_axilite port=comm_adr
#pragma HLS INTERFACE s_axilite port=dpcfg_adr
#pragma HLS INTERFACE m_axi port=table_HBM0 offset=slave bundle  = gmem7
#pragma HLS INTERFACE m_axi port=table_HBM1 offset=slave bundle  = gmem8
#pragma HLS INTERFACE m_axi port=table_HBM2 offset=slave bundle  = gmem9
#pragma HLS INTERFACE m_axi port=table_HBM3 offset=slave bundle  = gmem10
#pragma HLS INTERFACE m_axi port=table_HBM4 offset=slave bundle  = gmem11
#pragma HLS INTERFACE m_axi port=table_HBM5 offset=slave bundle  = gmem12
#pragma HLS INTERFACE m_axi port=table_HBM6 offset=slave bundle  = gmem13
#pragma HLS INTERFACE m_axi port=table_HBM7 offset=slave bundle  = gmem14

#pragma HLS INTERFACE axis port=cmd_to_cclo
#pragma HLS INTERFACE axis port=sts_from_cclo
#pragma HLS INTERFACE axis port=data_to_cclo
#pragma HLS INTERFACE axis port=data_from_cclo
#pragma HLS INTERFACE s_axilite port=return

    // Barrier
    accl_hls::barrier_non_root(
        local_rank,
        comm_size,
        comm_adr, 
        dpcfg_adr,
        cmd_to_cclo,
        sts_from_cclo,
        data_to_cclo,
        data_from_cclo
    );
    
    
    // // dlrm data flow
    // dlrm_embedding_compute(
    //     table_HBM0,
    //     table_HBM1,
    //     table_HBM2,
    //     table_HBM3,
    //     table_HBM4,
    //     table_HBM5,
    //     table_HBM6,
    //     table_HBM7,
    //     destination,
    //     comm_adr, 
    //     dpcfg_adr,
    //     cmd_to_cclo,
    //     sts_from_cclo,
    //     data_to_cclo,
    //     data_from_cclo);
    
}


// void dlrm_embedding(
//     int *src,
//     int count,
//     unsigned int destination,
//     //parameters pertaining to CCLO config
//     ap_uint<32> comm_adr, 
//     ap_uint<32> dpcfg_adr,
//     //streams to and from CCLO
//     STREAM<command_word> &cmd_to_cclo,
//     STREAM<command_word> &sts_from_cclo,
//     STREAM<stream_word> &data_to_cclo,
//     STREAM<stream_word> &data_from_cclo
// ){
// #pragma HLS INTERFACE s_axilite port=count
// #pragma HLS INTERFACE s_axilite port=destination
// #pragma HLS INTERFACE s_axilite port=comm_adr
// #pragma HLS INTERFACE s_axilite port=dpcfg_adr
// #pragma HLS INTERFACE m_axi port=src offset=slave
// #pragma HLS INTERFACE axis port=cmd_to_cclo
// #pragma HLS INTERFACE axis port=sts_from_cclo
// #pragma HLS INTERFACE axis port=data_to_cclo
// #pragma HLS INTERFACE axis port=data_from_cclo
// #pragma HLS INTERFACE s_axilite port=return

// #pragma HLS dataflow disable_start_propagation

//     //set up interfaces
//     accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);
//     accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
//     //send command to CCLO
//     //we're passing src as source and targeting stream 9
//     //because we're streaming data, the address will be ignored
//     accl.stream_put_nb(count, 9, destination, (ap_uint<64>)src); 
//     //read data from src
//     //and push the result into the CCLO stream
//     data.push_from_mem(src, count, 0);
//     data.tie_off_cclo2krnl();
//     #ifndef ACCL_SYNTHESIS
//         std::cout << "dlrm_embedding: stream_put count=" << count << " destination=" << destination << "\n";
//     #endif
//     accl.finalize_call();
//     #ifndef ACCL_SYNTHESIS
//         std::cout << "dlrm_embedding: finish" << "\n";
//     #endif
// }

