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
#include "compute_agg.hpp"

using namespace dlrm_agg_ns;

void dlrm_agg_compute(
    int *dst,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo,
    STREAM<int> &mem_wr_done
)
{
    STREAM<ap_uint<512> > s_result2_partial;
#pragma HLS stream variable=s_result2_partial depth=512
    STREAM<ap_uint<512> > s_feature_in_layer_2;
#pragma HLS stream variable=s_feature_in_layer_2 depth=512
    STREAM<ap_uint<512> > s_feature_out_layer_2;
#pragma HLS stream variable=s_feature_out_layer_2 depth=512
    STREAM<ap_uint<512> > s_feature_in_layer_3;
#pragma HLS stream variable=s_feature_in_layer_3 depth=512
    STREAM<ap_uint<512> > s_feature_out_layer_3;
#pragma HLS stream variable=s_feature_out_layer_3 depth=512


    STREAM<W_TYPE> s_feature2_PE0_0;
    STREAM<W_TYPE> s_feature2_PE0_1;
    STREAM<D_TYPE> s_result2_PE0;
#pragma HLS stream variable=s_feature2_PE0_0 depth=2
#pragma HLS stream variable=s_feature2_PE0_1 depth=2
#pragma HLS stream variable=s_result2_PE0 depth=2
    STREAM<W_TYPE> s_feature2_PE1_0;
    STREAM<W_TYPE> s_feature2_PE1_1;
    STREAM<D_TYPE> s_result2_PE1;
#pragma HLS stream variable=s_feature2_PE1_0 depth=2
#pragma HLS stream variable=s_feature2_PE1_1 depth=2
#pragma HLS stream variable=s_result2_PE1 depth=2
    STREAM<W_TYPE> s_feature2_PE2_0;
    STREAM<W_TYPE> s_feature2_PE2_1;
    STREAM<D_TYPE> s_result2_PE2;
#pragma HLS stream variable=s_feature2_PE2_0 depth=2
#pragma HLS stream variable=s_feature2_PE2_1 depth=2
#pragma HLS stream variable=s_result2_PE2 depth=2
    STREAM<W_TYPE> s_feature2_PE3_0;
    STREAM<W_TYPE> s_feature2_PE3_1;
    STREAM<D_TYPE> s_result2_PE3;
#pragma HLS stream variable=s_feature2_PE3_0 depth=2
#pragma HLS stream variable=s_feature2_PE3_1 depth=2
#pragma HLS stream variable=s_result2_PE3 depth=2
    STREAM<W_TYPE> s_feature2_PE4_0;
    STREAM<W_TYPE> s_feature2_PE4_1;
    STREAM<D_TYPE> s_result2_PE4;
#pragma HLS stream variable=s_feature2_PE4_0 depth=2
#pragma HLS stream variable=s_feature2_PE4_1 depth=2
#pragma HLS stream variable=s_result2_PE4 depth=2
    STREAM<W_TYPE> s_feature2_PE5_0;
    STREAM<W_TYPE> s_feature2_PE5_1;
    STREAM<D_TYPE> s_result2_PE5;
#pragma HLS stream variable=s_feature2_PE5_0 depth=2
#pragma HLS stream variable=s_feature2_PE5_1 depth=2
#pragma HLS stream variable=s_result2_PE5 depth=2
    STREAM<W_TYPE> s_feature2_PE6_0;
    STREAM<W_TYPE> s_feature2_PE6_1;
    STREAM<D_TYPE> s_result2_PE6;
#pragma HLS stream variable=s_feature2_PE6_0 depth=2
#pragma HLS stream variable=s_feature2_PE6_1 depth=2
#pragma HLS stream variable=s_result2_PE6 depth=2
    STREAM<W_TYPE> s_feature2_PE7_0;
    STREAM<W_TYPE> s_feature2_PE7_1;
    STREAM<D_TYPE> s_result2_PE7;
#pragma HLS stream variable=s_feature2_PE7_0 depth=2
#pragma HLS stream variable=s_feature2_PE7_1 depth=2
#pragma HLS stream variable=s_result2_PE7 depth=2
    STREAM<W_TYPE> s_feature2_PE8_0;
    STREAM<W_TYPE> s_feature2_PE8_1;
    STREAM<D_TYPE> s_result2_PE8;
#pragma HLS stream variable=s_feature2_PE8_0 depth=2
#pragma HLS stream variable=s_feature2_PE8_1 depth=2
#pragma HLS stream variable=s_result2_PE8 depth=2
    STREAM<W_TYPE> s_feature2_PE9_0;
    STREAM<W_TYPE> s_feature2_PE9_1;
    STREAM<D_TYPE> s_result2_PE9;
#pragma HLS stream variable=s_feature2_PE9_0 depth=2
#pragma HLS stream variable=s_feature2_PE9_1 depth=2
#pragma HLS stream variable=s_result2_PE9 depth=2
    STREAM<W_TYPE> s_feature2_PE10_0;
    STREAM<W_TYPE> s_feature2_PE10_1;
    STREAM<D_TYPE> s_result2_PE10;
#pragma HLS stream variable=s_feature2_PE10_0 depth=2
#pragma HLS stream variable=s_feature2_PE10_1 depth=2
#pragma HLS stream variable=s_result2_PE10 depth=2
    STREAM<W_TYPE> s_feature2_PE11_0;
    STREAM<W_TYPE> s_feature2_PE11_1;
    STREAM<D_TYPE> s_result2_PE11;
#pragma HLS stream variable=s_feature2_PE11_0 depth=2
#pragma HLS stream variable=s_feature2_PE11_1 depth=2
#pragma HLS stream variable=s_result2_PE11 depth=2
    STREAM<W_TYPE> s_feature2_PE12_0;
    STREAM<W_TYPE> s_feature2_PE12_1;
    STREAM<D_TYPE> s_result2_PE12;
#pragma HLS stream variable=s_feature2_PE12_0 depth=2
#pragma HLS stream variable=s_feature2_PE12_1 depth=2
#pragma HLS stream variable=s_result2_PE12 depth=2
    STREAM<W_TYPE> s_feature2_PE13_0;
    STREAM<W_TYPE> s_feature2_PE13_1;
    STREAM<D_TYPE> s_result2_PE13;
#pragma HLS stream variable=s_feature2_PE13_0 depth=2
#pragma HLS stream variable=s_feature2_PE13_1 depth=2
#pragma HLS stream variable=s_result2_PE13 depth=2
    STREAM<W_TYPE> s_feature2_PE14_0;
    STREAM<W_TYPE> s_feature2_PE14_1;
    STREAM<D_TYPE> s_result2_PE14;
#pragma HLS stream variable=s_feature2_PE14_0 depth=2
#pragma HLS stream variable=s_feature2_PE14_1 depth=2
#pragma HLS stream variable=s_result2_PE14 depth=2
    STREAM<W_TYPE> s_feature2_PE15_0;
    STREAM<W_TYPE> s_feature2_PE15_1;
    STREAM<D_TYPE> s_result2_PE15;
#pragma HLS stream variable=s_feature2_PE15_0 depth=2
#pragma HLS stream variable=s_feature2_PE15_1 depth=2
#pragma HLS stream variable=s_result2_PE15 depth=2
    STREAM<W_TYPE> s_feature2_PE16_0;
    STREAM<W_TYPE> s_feature2_PE16_1;
    STREAM<D_TYPE> s_result2_PE16;
#pragma HLS stream variable=s_feature2_PE16_0 depth=2
#pragma HLS stream variable=s_feature2_PE16_1 depth=2
#pragma HLS stream variable=s_result2_PE16 depth=2
    STREAM<W_TYPE> s_feature2_PE17_0;
    STREAM<W_TYPE> s_feature2_PE17_1;
    STREAM<D_TYPE> s_result2_PE17;
#pragma HLS stream variable=s_feature2_PE17_0 depth=2
#pragma HLS stream variable=s_feature2_PE17_1 depth=2
#pragma HLS stream variable=s_result2_PE17 depth=2
    STREAM<W_TYPE> s_feature2_PE18_0;
    STREAM<W_TYPE> s_feature2_PE18_1;
    STREAM<D_TYPE> s_result2_PE18;
#pragma HLS stream variable=s_feature2_PE18_0 depth=2
#pragma HLS stream variable=s_feature2_PE18_1 depth=2
#pragma HLS stream variable=s_result2_PE18 depth=2
    STREAM<W_TYPE> s_feature2_PE19_0;
    STREAM<W_TYPE> s_feature2_PE19_1;
    STREAM<D_TYPE> s_result2_PE19;
#pragma HLS stream variable=s_feature2_PE19_0 depth=2
#pragma HLS stream variable=s_feature2_PE19_1 depth=2
#pragma HLS stream variable=s_result2_PE19 depth=2
    STREAM<W_TYPE> s_feature2_PE20_0;
    STREAM<W_TYPE> s_feature2_PE20_1;
    STREAM<D_TYPE> s_result2_PE20;
#pragma HLS stream variable=s_feature2_PE20_0 depth=2
#pragma HLS stream variable=s_feature2_PE20_1 depth=2
#pragma HLS stream variable=s_result2_PE20 depth=2
    STREAM<W_TYPE> s_feature2_PE21_0;
    STREAM<W_TYPE> s_feature2_PE21_1;
    STREAM<D_TYPE> s_result2_PE21;
#pragma HLS stream variable=s_feature2_PE21_0 depth=2
#pragma HLS stream variable=s_feature2_PE21_1 depth=2
#pragma HLS stream variable=s_result2_PE21 depth=2
    STREAM<W_TYPE> s_feature2_PE22_0;
    STREAM<W_TYPE> s_feature2_PE22_1;
    STREAM<D_TYPE> s_result2_PE22;
#pragma HLS stream variable=s_feature2_PE22_0 depth=2
#pragma HLS stream variable=s_feature2_PE22_1 depth=2
#pragma HLS stream variable=s_result2_PE22 depth=2
    STREAM<W_TYPE> s_feature2_PE23_0;
    STREAM<W_TYPE> s_feature2_PE23_1;
    STREAM<D_TYPE> s_result2_PE23;
#pragma HLS stream variable=s_feature2_PE23_0 depth=2
#pragma HLS stream variable=s_feature2_PE23_1 depth=2
#pragma HLS stream variable=s_result2_PE23 depth=2
    STREAM<W_TYPE> s_feature2_PE24_0;
    STREAM<W_TYPE> s_feature2_PE24_1;
    STREAM<D_TYPE> s_result2_PE24;
#pragma HLS stream variable=s_feature2_PE24_0 depth=2
#pragma HLS stream variable=s_feature2_PE24_1 depth=2
#pragma HLS stream variable=s_result2_PE24 depth=2
    STREAM<W_TYPE> s_feature2_PE25_0;
    STREAM<W_TYPE> s_feature2_PE25_1;
    STREAM<D_TYPE> s_result2_PE25;
#pragma HLS stream variable=s_feature2_PE25_0 depth=2
#pragma HLS stream variable=s_feature2_PE25_1 depth=2
#pragma HLS stream variable=s_result2_PE25 depth=2
    STREAM<W_TYPE> s_feature2_PE26_0;
    STREAM<W_TYPE> s_feature2_PE26_1;
    STREAM<D_TYPE> s_result2_PE26;
#pragma HLS stream variable=s_feature2_PE26_0 depth=2
#pragma HLS stream variable=s_feature2_PE26_1 depth=2
#pragma HLS stream variable=s_result2_PE26 depth=2
    STREAM<W_TYPE> s_feature2_PE27_0;
    STREAM<W_TYPE> s_feature2_PE27_1;
    STREAM<D_TYPE> s_result2_PE27;
#pragma HLS stream variable=s_feature2_PE27_0 depth=2
#pragma HLS stream variable=s_feature2_PE27_1 depth=2
#pragma HLS stream variable=s_result2_PE27 depth=2
    STREAM<W_TYPE> s_feature2_PE28_0;
    STREAM<W_TYPE> s_feature2_PE28_1;
    STREAM<D_TYPE> s_result2_PE28;
#pragma HLS stream variable=s_feature2_PE28_0 depth=2
#pragma HLS stream variable=s_feature2_PE28_1 depth=2
#pragma HLS stream variable=s_result2_PE28 depth=2
    STREAM<W_TYPE> s_feature2_PE29_0;
    STREAM<W_TYPE> s_feature2_PE29_1;
    STREAM<D_TYPE> s_result2_PE29;
#pragma HLS stream variable=s_feature2_PE29_0 depth=2
#pragma HLS stream variable=s_feature2_PE29_1 depth=2
#pragma HLS stream variable=s_result2_PE29 depth=2
    STREAM<W_TYPE> s_feature2_PE30_0;
    STREAM<W_TYPE> s_feature2_PE30_1;
    STREAM<D_TYPE> s_result2_PE30;
#pragma HLS stream variable=s_feature2_PE30_0 depth=2
#pragma HLS stream variable=s_feature2_PE30_1 depth=2
#pragma HLS stream variable=s_result2_PE30 depth=2
    STREAM<W_TYPE> s_feature2_PE31_0;
    STREAM<W_TYPE> s_feature2_PE31_1;
    STREAM<D_TYPE> s_result2_PE31;
#pragma HLS stream variable=s_feature2_PE31_0 depth=2
#pragma HLS stream variable=s_feature2_PE31_1 depth=2
#pragma HLS stream variable=s_result2_PE31 depth=2

    STREAM<ap_uint<512> > s_result2_partial_0;
#pragma HLS stream variable=s_result2_partial_0 depth=128



    STREAM<W_TYPE> s_feature3_PE0_0;
    STREAM<W_TYPE> s_feature3_PE0_1;
    STREAM<D_TYPE> s_result3_PE0;
#pragma HLS stream variable=s_feature3_PE0_0 depth=2
#pragma HLS stream variable=s_feature3_PE0_1 depth=2
#pragma HLS stream variable=s_result3_PE0 depth=2
    STREAM<W_TYPE> s_feature3_PE1_0;
    STREAM<W_TYPE> s_feature3_PE1_1;
    STREAM<D_TYPE> s_result3_PE1;
#pragma HLS stream variable=s_feature3_PE1_0 depth=2
#pragma HLS stream variable=s_feature3_PE1_1 depth=2
#pragma HLS stream variable=s_result3_PE1 depth=2
    STREAM<W_TYPE> s_feature3_PE2_0;
    STREAM<W_TYPE> s_feature3_PE2_1;
    STREAM<D_TYPE> s_result3_PE2;
#pragma HLS stream variable=s_feature3_PE2_0 depth=2
#pragma HLS stream variable=s_feature3_PE2_1 depth=2
#pragma HLS stream variable=s_result3_PE2 depth=2
    STREAM<W_TYPE> s_feature3_PE3_0;
    STREAM<W_TYPE> s_feature3_PE3_1;
    STREAM<D_TYPE> s_result3_PE3;
#pragma HLS stream variable=s_feature3_PE3_0 depth=2
#pragma HLS stream variable=s_feature3_PE3_1 depth=2
#pragma HLS stream variable=s_result3_PE3 depth=2
    STREAM<W_TYPE> s_feature3_PE4_0;
    STREAM<W_TYPE> s_feature3_PE4_1;
    STREAM<D_TYPE> s_result3_PE4;
#pragma HLS stream variable=s_feature3_PE4_0 depth=2
#pragma HLS stream variable=s_feature3_PE4_1 depth=2
#pragma HLS stream variable=s_result3_PE4 depth=2
    STREAM<W_TYPE> s_feature3_PE5_0;
    STREAM<W_TYPE> s_feature3_PE5_1;
    STREAM<D_TYPE> s_result3_PE5;
#pragma HLS stream variable=s_feature3_PE5_0 depth=2
#pragma HLS stream variable=s_feature3_PE5_1 depth=2
#pragma HLS stream variable=s_result3_PE5 depth=2
    STREAM<W_TYPE> s_feature3_PE6_0;
    STREAM<W_TYPE> s_feature3_PE6_1;
    STREAM<D_TYPE> s_result3_PE6;
#pragma HLS stream variable=s_feature3_PE6_0 depth=2
#pragma HLS stream variable=s_feature3_PE6_1 depth=2
#pragma HLS stream variable=s_result3_PE6 depth=2
    STREAM<W_TYPE> s_feature3_PE7_0;
    STREAM<W_TYPE> s_feature3_PE7_1;
    STREAM<D_TYPE> s_result3_PE7;
#pragma HLS stream variable=s_feature3_PE7_0 depth=2
#pragma HLS stream variable=s_feature3_PE7_1 depth=2
#pragma HLS stream variable=s_result3_PE7 depth=2
    STREAM<W_TYPE> s_feature3_PE8_0;
    STREAM<W_TYPE> s_feature3_PE8_1;
    STREAM<D_TYPE> s_result3_PE8;
#pragma HLS stream variable=s_feature3_PE8_0 depth=2
#pragma HLS stream variable=s_feature3_PE8_1 depth=2
#pragma HLS stream variable=s_result3_PE8 depth=2
    STREAM<W_TYPE> s_feature3_PE9_0;
    STREAM<W_TYPE> s_feature3_PE9_1;
    STREAM<D_TYPE> s_result3_PE9;
#pragma HLS stream variable=s_feature3_PE9_0 depth=2
#pragma HLS stream variable=s_feature3_PE9_1 depth=2
#pragma HLS stream variable=s_result3_PE9 depth=2
    STREAM<W_TYPE> s_feature3_PE10_0;
    STREAM<W_TYPE> s_feature3_PE10_1;
    STREAM<D_TYPE> s_result3_PE10;
#pragma HLS stream variable=s_feature3_PE10_0 depth=2
#pragma HLS stream variable=s_feature3_PE10_1 depth=2
#pragma HLS stream variable=s_result3_PE10 depth=2
    STREAM<W_TYPE> s_feature3_PE11_0;
    STREAM<W_TYPE> s_feature3_PE11_1;
    STREAM<D_TYPE> s_result3_PE11;
#pragma HLS stream variable=s_feature3_PE11_0 depth=2
#pragma HLS stream variable=s_feature3_PE11_1 depth=2
#pragma HLS stream variable=s_result3_PE11 depth=2
    STREAM<W_TYPE> s_feature3_PE12_0;
    STREAM<W_TYPE> s_feature3_PE12_1;
    STREAM<D_TYPE> s_result3_PE12;
#pragma HLS stream variable=s_feature3_PE12_0 depth=2
#pragma HLS stream variable=s_feature3_PE12_1 depth=2
#pragma HLS stream variable=s_result3_PE12 depth=2
    STREAM<W_TYPE> s_feature3_PE13_0;
    STREAM<W_TYPE> s_feature3_PE13_1;
    STREAM<D_TYPE> s_result3_PE13;
#pragma HLS stream variable=s_feature3_PE13_0 depth=2
#pragma HLS stream variable=s_feature3_PE13_1 depth=2
#pragma HLS stream variable=s_result3_PE13 depth=2
    STREAM<W_TYPE> s_feature3_PE14_0;
    STREAM<W_TYPE> s_feature3_PE14_1;
    STREAM<D_TYPE> s_result3_PE14;
#pragma HLS stream variable=s_feature3_PE14_0 depth=2
#pragma HLS stream variable=s_feature3_PE14_1 depth=2
#pragma HLS stream variable=s_result3_PE14 depth=2
    STREAM<W_TYPE> s_feature3_PE15_0;
    STREAM<W_TYPE> s_feature3_PE15_1;
    STREAM<D_TYPE> s_result3_PE15;
#pragma HLS stream variable=s_feature3_PE15_0 depth=2
#pragma HLS stream variable=s_feature3_PE15_1 depth=2
#pragma HLS stream variable=s_result3_PE15 depth=2
    STREAM<W_TYPE> s_feature3_PE16_0;
    STREAM<W_TYPE> s_feature3_PE16_1;
    STREAM<D_TYPE> s_result3_PE16;
#pragma HLS stream variable=s_feature3_PE16_0 depth=2
#pragma HLS stream variable=s_feature3_PE16_1 depth=2
#pragma HLS stream variable=s_result3_PE16 depth=2
    STREAM<W_TYPE> s_feature3_PE17_0;
    STREAM<W_TYPE> s_feature3_PE17_1;
    STREAM<D_TYPE> s_result3_PE17;
#pragma HLS stream variable=s_feature3_PE17_0 depth=2
#pragma HLS stream variable=s_feature3_PE17_1 depth=2
#pragma HLS stream variable=s_result3_PE17 depth=2
    STREAM<W_TYPE> s_feature3_PE18_0;
    STREAM<W_TYPE> s_feature3_PE18_1;
    STREAM<D_TYPE> s_result3_PE18;
#pragma HLS stream variable=s_feature3_PE18_0 depth=2
#pragma HLS stream variable=s_feature3_PE18_1 depth=2
#pragma HLS stream variable=s_result3_PE18 depth=2
    STREAM<W_TYPE> s_feature3_PE19_0;
    STREAM<W_TYPE> s_feature3_PE19_1;
    STREAM<D_TYPE> s_result3_PE19;
#pragma HLS stream variable=s_feature3_PE19_0 depth=2
#pragma HLS stream variable=s_feature3_PE19_1 depth=2
#pragma HLS stream variable=s_result3_PE19 depth=2
    STREAM<W_TYPE> s_feature3_PE20_0;
    STREAM<W_TYPE> s_feature3_PE20_1;
    STREAM<D_TYPE> s_result3_PE20;
#pragma HLS stream variable=s_feature3_PE20_0 depth=2
#pragma HLS stream variable=s_feature3_PE20_1 depth=2
#pragma HLS stream variable=s_result3_PE20 depth=2
    STREAM<W_TYPE> s_feature3_PE21_0;
    STREAM<W_TYPE> s_feature3_PE21_1;
    STREAM<D_TYPE> s_result3_PE21;
#pragma HLS stream variable=s_feature3_PE21_0 depth=2
#pragma HLS stream variable=s_feature3_PE21_1 depth=2
#pragma HLS stream variable=s_result3_PE21 depth=2
    STREAM<W_TYPE> s_feature3_PE22_0;
    STREAM<W_TYPE> s_feature3_PE22_1;
    STREAM<D_TYPE> s_result3_PE22;
#pragma HLS stream variable=s_feature3_PE22_0 depth=2
#pragma HLS stream variable=s_feature3_PE22_1 depth=2
#pragma HLS stream variable=s_result3_PE22 depth=2
    STREAM<W_TYPE> s_feature3_PE23_0;
    STREAM<W_TYPE> s_feature3_PE23_1;
    STREAM<D_TYPE> s_result3_PE23;
#pragma HLS stream variable=s_feature3_PE23_0 depth=2
#pragma HLS stream variable=s_feature3_PE23_1 depth=2
#pragma HLS stream variable=s_result3_PE23 depth=2
    STREAM<W_TYPE> s_feature3_PE24_0;
    STREAM<W_TYPE> s_feature3_PE24_1;
    STREAM<D_TYPE> s_result3_PE24;
#pragma HLS stream variable=s_feature3_PE24_0 depth=2
#pragma HLS stream variable=s_feature3_PE24_1 depth=2
#pragma HLS stream variable=s_result3_PE24 depth=2
    STREAM<W_TYPE> s_feature3_PE25_0;
    STREAM<W_TYPE> s_feature3_PE25_1;
    STREAM<D_TYPE> s_result3_PE25;
#pragma HLS stream variable=s_feature3_PE25_0 depth=2
#pragma HLS stream variable=s_feature3_PE25_1 depth=2
#pragma HLS stream variable=s_result3_PE25 depth=2
    STREAM<W_TYPE> s_feature3_PE26_0;
    STREAM<W_TYPE> s_feature3_PE26_1;
    STREAM<D_TYPE> s_result3_PE26;
#pragma HLS stream variable=s_feature3_PE26_0 depth=2
#pragma HLS stream variable=s_feature3_PE26_1 depth=2
#pragma HLS stream variable=s_result3_PE26 depth=2
    STREAM<W_TYPE> s_feature3_PE27_0;
    STREAM<W_TYPE> s_feature3_PE27_1;
    STREAM<D_TYPE> s_result3_PE27;
#pragma HLS stream variable=s_feature3_PE27_0 depth=2
#pragma HLS stream variable=s_feature3_PE27_1 depth=2
#pragma HLS stream variable=s_result3_PE27 depth=2
    STREAM<W_TYPE> s_feature3_PE28_0;
    STREAM<W_TYPE> s_feature3_PE28_1;
    STREAM<D_TYPE> s_result3_PE28;
#pragma HLS stream variable=s_feature3_PE28_0 depth=2
#pragma HLS stream variable=s_feature3_PE28_1 depth=2
#pragma HLS stream variable=s_result3_PE28 depth=2
    STREAM<W_TYPE> s_feature3_PE29_0;
    STREAM<W_TYPE> s_feature3_PE29_1;
    STREAM<D_TYPE> s_result3_PE29;
#pragma HLS stream variable=s_feature3_PE29_0 depth=2
#pragma HLS stream variable=s_feature3_PE29_1 depth=2
#pragma HLS stream variable=s_result3_PE29 depth=2
    STREAM<W_TYPE> s_feature3_PE30_0;
    STREAM<W_TYPE> s_feature3_PE30_1;
    STREAM<D_TYPE> s_result3_PE30;
#pragma HLS stream variable=s_feature3_PE30_0 depth=2
#pragma HLS stream variable=s_feature3_PE30_1 depth=2
#pragma HLS stream variable=s_result3_PE30 depth=2
    STREAM<W_TYPE> s_feature3_PE31_0;
    STREAM<W_TYPE> s_feature3_PE31_1;
    STREAM<D_TYPE> s_result3_PE31;
#pragma HLS stream variable=s_feature3_PE31_0 depth=2
#pragma HLS stream variable=s_feature3_PE31_1 depth=2
#pragma HLS stream variable=s_result3_PE31 depth=2

    STREAM<W_TYPE> s_result3_partial_0;
#pragma HLS stream variable=s_result3_partial_0 depth=128


    STREAM<W_TYPE> s_result3_all;
#pragma HLS stream variable=s_result3_all depth=256

    STREAM<D_TYPE> s_result_out;
#pragma HLS stream variable=s_result_out depth=256

    STREAM<ap_uint<512> > s_data_in_zero;
#pragma HLS stream variable=s_data_in_zero depth=256

static STREAM<ap_uint<512> > s_data_in;
#pragma HLS STREAM variable=s_data_in depth=512

static STREAM<ap_uint<512> >    s_data_out;
#pragma HLS STREAM variable=s_data_out depth=512

#pragma HLS dataflow disable_start_propagation

    int count_recv = BATCH_NUM * BATCH_SIZE * (2 * 64 + 16) * 16;
    //set up interfaces
    // accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    //pull data from CCLO and write it to dst
    // data.pull_to_mem(dst, count);
    data.pull_to_stream(s_data_in, count_recv);

    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_agg: recv count:"<< count_recv << "\n";
    #endif

    //s_result2_partial 16 words, s_feature_in_layer_2 128 words 
    recvDataTransform(s_data_in, s_result2_partial, s_feature_in_layer_2);


/////////////////////////////////////////////////////////////////////////////
    store_features<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature_in_layer_2, s_feature_out_layer_2);

    replicate_feature_512PEs_32PE<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(
        s_feature_out_layer_2, 
        s_feature2_PE0_0, s_feature2_PE0_1, s_feature2_PE1_0, s_feature2_PE1_1,
        s_feature2_PE2_0, s_feature2_PE2_1, s_feature2_PE3_0, s_feature2_PE3_1,
        s_feature2_PE4_0, s_feature2_PE4_1, s_feature2_PE5_0, s_feature2_PE5_1,
        s_feature2_PE6_0, s_feature2_PE6_1, s_feature2_PE7_0, s_feature2_PE7_1,
        s_feature2_PE8_0, s_feature2_PE8_1, s_feature2_PE9_0, s_feature2_PE9_1,
        s_feature2_PE10_0, s_feature2_PE10_1, s_feature2_PE11_0, s_feature2_PE11_1,
        s_feature2_PE12_0, s_feature2_PE12_1, s_feature2_PE13_0, s_feature2_PE13_1,
        s_feature2_PE14_0, s_feature2_PE14_1, s_feature2_PE15_0, s_feature2_PE15_1,
        s_feature2_PE16_0, s_feature2_PE16_1, s_feature2_PE17_0, s_feature2_PE17_1,
        s_feature2_PE18_0, s_feature2_PE18_1, s_feature2_PE19_0, s_feature2_PE19_1,
        s_feature2_PE20_0, s_feature2_PE20_1, s_feature2_PE21_0, s_feature2_PE21_1,
        s_feature2_PE22_0, s_feature2_PE22_1, s_feature2_PE23_0, s_feature2_PE23_1,
        s_feature2_PE24_0, s_feature2_PE24_1, s_feature2_PE25_0, s_feature2_PE25_1,
        s_feature2_PE26_0, s_feature2_PE26_1, s_feature2_PE27_0, s_feature2_PE27_1,
        s_feature2_PE28_0, s_feature2_PE28_1, s_feature2_PE29_0, s_feature2_PE29_1,
        s_feature2_PE30_0, s_feature2_PE30_1, s_feature2_PE31_0, s_feature2_PE31_1);

    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE0_0, s_feature2_PE0_1, s_result2_PE0);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE1_0, s_feature2_PE1_1, s_result2_PE1);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE2_0, s_feature2_PE2_1, s_result2_PE2);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE3_0, s_feature2_PE3_1, s_result2_PE3);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE4_0, s_feature2_PE4_1, s_result2_PE4);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE5_0, s_feature2_PE5_1, s_result2_PE5);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE6_0, s_feature2_PE6_1, s_result2_PE6);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE7_0, s_feature2_PE7_1, s_result2_PE7);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE8_0, s_feature2_PE8_1, s_result2_PE8);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE9_0, s_feature2_PE9_1, s_result2_PE9);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE10_0, s_feature2_PE10_1, s_result2_PE10);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE11_0, s_feature2_PE11_1, s_result2_PE11);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE12_0, s_feature2_PE12_1, s_result2_PE12);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE13_0, s_feature2_PE13_1, s_result2_PE13);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE14_0, s_feature2_PE14_1, s_result2_PE14);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE15_0, s_feature2_PE15_1, s_result2_PE15);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE16_0, s_feature2_PE16_1, s_result2_PE16);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE17_0, s_feature2_PE17_1, s_result2_PE17);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE18_0, s_feature2_PE18_1, s_result2_PE18);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE19_0, s_feature2_PE19_1, s_result2_PE19);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE20_0, s_feature2_PE20_1, s_result2_PE20);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE21_0, s_feature2_PE21_1, s_result2_PE21);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE22_0, s_feature2_PE22_1, s_result2_PE22);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE23_0, s_feature2_PE23_1, s_result2_PE23);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE24_0, s_feature2_PE24_1, s_result2_PE24);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE25_0, s_feature2_PE25_1, s_result2_PE25);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE26_0, s_feature2_PE26_1, s_result2_PE26);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE27_0, s_feature2_PE27_1, s_result2_PE27);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE28_0, s_feature2_PE28_1, s_result2_PE28);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE29_0, s_feature2_PE29_1, s_result2_PE29);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE30_0, s_feature2_PE30_1, s_result2_PE30);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE1, AGG_ROW_PER_PE2>(s_feature2_PE31_0, s_feature2_PE31_1, s_result2_PE31);

    gather_results_32PEs_layer_2<AGG_ROW_PER_PE2>(
        s_result2_PE0, s_result2_PE1, s_result2_PE2, s_result2_PE3,
        s_result2_PE4, s_result2_PE5, s_result2_PE6, s_result2_PE7,
        s_result2_PE8, s_result2_PE9, s_result2_PE10, s_result2_PE11,
        s_result2_PE12, s_result2_PE13, s_result2_PE14, s_result2_PE15,
        s_result2_PE16, s_result2_PE17, s_result2_PE18, s_result2_PE19,
        s_result2_PE20, s_result2_PE21, s_result2_PE22, s_result2_PE23,
        s_result2_PE24, s_result2_PE25, s_result2_PE26, s_result2_PE27,
        s_result2_PE28, s_result2_PE29, s_result2_PE30, s_result2_PE31,
        s_result2_partial_0);

    // s_result2_partial is 16 words, s_result2_partial_0 is 2 words, pad 14 words
    // output is 32 words
    gather_results_layer_2<AGG_ROW_PER_PE2>(
        s_result2_partial_0, 
        s_result2_partial,
        s_feature_in_layer_3);
    

    // Compute layer 3
    // s_feature_in_layer_3 consume 32 words
    store_features<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature_in_layer_3, s_feature_out_layer_3);

    replicate_feature_512PEs_32PE<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(
        s_feature_out_layer_3, 
        s_feature3_PE0_0, s_feature3_PE0_1, s_feature3_PE1_0, s_feature3_PE1_1,
        s_feature3_PE2_0, s_feature3_PE2_1, s_feature3_PE3_0, s_feature3_PE3_1,
        s_feature3_PE4_0, s_feature3_PE4_1, s_feature3_PE5_0, s_feature3_PE5_1,
        s_feature3_PE6_0, s_feature3_PE6_1, s_feature3_PE7_0, s_feature3_PE7_1,
        s_feature3_PE8_0, s_feature3_PE8_1, s_feature3_PE9_0, s_feature3_PE9_1,
        s_feature3_PE10_0, s_feature3_PE10_1, s_feature3_PE11_0, s_feature3_PE11_1,
        s_feature3_PE12_0, s_feature3_PE12_1, s_feature3_PE13_0, s_feature3_PE13_1,
        s_feature3_PE14_0, s_feature3_PE14_1, s_feature3_PE15_0, s_feature3_PE15_1,
        s_feature3_PE16_0, s_feature3_PE16_1, s_feature3_PE17_0, s_feature3_PE17_1,
        s_feature3_PE18_0, s_feature3_PE18_1, s_feature3_PE19_0, s_feature3_PE19_1,
        s_feature3_PE20_0, s_feature3_PE20_1, s_feature3_PE21_0, s_feature3_PE21_1,
        s_feature3_PE22_0, s_feature3_PE22_1, s_feature3_PE23_0, s_feature3_PE23_1,
        s_feature3_PE24_0, s_feature3_PE24_1, s_feature3_PE25_0, s_feature3_PE25_1,
        s_feature3_PE26_0, s_feature3_PE26_1, s_feature3_PE27_0, s_feature3_PE27_1,
        s_feature3_PE28_0, s_feature3_PE28_1, s_feature3_PE29_0, s_feature3_PE29_1,
        s_feature3_PE30_0, s_feature3_PE30_1, s_feature3_PE31_0, s_feature3_PE31_1);
                

    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE0_0, s_feature3_PE0_1, s_result3_PE0);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE1_0, s_feature3_PE1_1, s_result3_PE1);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE2_0, s_feature3_PE2_1, s_result3_PE2);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE3_0, s_feature3_PE3_1, s_result3_PE3);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE4_0, s_feature3_PE4_1, s_result3_PE4);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE5_0, s_feature3_PE5_1, s_result3_PE5);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE6_0, s_feature3_PE6_1, s_result3_PE6);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE7_0, s_feature3_PE7_1, s_result3_PE7);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE8_0, s_feature3_PE8_1, s_result3_PE8);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE9_0, s_feature3_PE9_1, s_result3_PE9);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE10_0, s_feature3_PE10_1, s_result3_PE10);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE11_0, s_feature3_PE11_1, s_result3_PE11);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE12_0, s_feature3_PE12_1, s_result3_PE12);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE13_0, s_feature3_PE13_1, s_result3_PE13);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE14_0, s_feature3_PE14_1, s_result3_PE14);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE15_0, s_feature3_PE15_1, s_result3_PE15);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE16_0, s_feature3_PE16_1, s_result3_PE16);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE17_0, s_feature3_PE17_1, s_result3_PE17);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE18_0, s_feature3_PE18_1, s_result3_PE18);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE19_0, s_feature3_PE19_1, s_result3_PE19);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE20_0, s_feature3_PE20_1, s_result3_PE20);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE21_0, s_feature3_PE21_1, s_result3_PE21);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE22_0, s_feature3_PE22_1, s_result3_PE22);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE23_0, s_feature3_PE23_1, s_result3_PE23);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE24_0, s_feature3_PE24_1, s_result3_PE24);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE25_0, s_feature3_PE25_1, s_result3_PE25);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE26_0, s_feature3_PE26_1, s_result3_PE26);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE27_0, s_feature3_PE27_1, s_result3_PE27);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE28_0, s_feature3_PE28_1, s_result3_PE28);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE29_0, s_feature3_PE29_1, s_result3_PE29);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE30_0, s_feature3_PE30_1, s_result3_PE30);
    matmul_PE_UNROLL8<AGG_HIDDEN_SIZE2, AGG_ROW_PER_PE3>(s_feature3_PE31_0, s_feature3_PE31_1, s_result3_PE31);

    gather_results_32PEs_layer_3<AGG_ROW_PER_PE3>(
        s_result3_PE0, s_result3_PE1, s_result3_PE2, s_result3_PE3,
        s_result3_PE4, s_result3_PE5, s_result3_PE6, s_result3_PE7,
        s_result3_PE8, s_result3_PE9, s_result3_PE10, s_result3_PE11,
        s_result3_PE12, s_result3_PE13, s_result3_PE14, s_result3_PE15,
        s_result3_PE16, s_result3_PE17, s_result3_PE18, s_result3_PE19,
        s_result3_PE20, s_result3_PE21, s_result3_PE22, s_result3_PE23,
        s_result3_PE24, s_result3_PE25, s_result3_PE26, s_result3_PE27,
        s_result3_PE28, s_result3_PE29, s_result3_PE30, s_result3_PE31,
        s_result3_partial_0);

    gather_results_layer_3<AGG_ROW_PER_PE3>(
        s_result3_partial_0, 
        s_result3_all);

    output_layer(s_result3_all, s_result_out);

    // dataTransform(s_result_out, s_data_out);
    write_to_mem(s_result_out, dst, mem_wr_done);


    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_agg: finish" << "\n";
    #endif
}

void check_done(
    STREAM<int> &mem_wr_done,
    ap_uint<32> comm_adr, 
    ap_uint<32> dpcfg_adr,
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo
)
{
#pragma HLS INLINE off

    int mem_wr_cnt = mem_wr_done.read();
    //send out a nop for measurement purposes
    accl_hls::start(ACCL_NOP, mem_wr_cnt, comm_adr, 0, 0, 0, dpcfg_adr, 0, 0, 0, 0, 0, cmd_to_cclo);
    accl_hls::finalize(sts_from_cclo);

    #ifndef ACCL_SYNTHESIS
        std::cout << "dlrm_agg NOP finish" << "\n";
    #endif
}

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
){
#pragma HLS INTERFACE s_axilite port=local_rank
#pragma HLS INTERFACE s_axilite port=comm_size
#pragma HLS INTERFACE s_axilite port=comm_adr
#pragma HLS INTERFACE s_axilite port=dpcfg_adr
#pragma HLS INTERFACE m_axi port=dst depth=1600 offset=slave
#pragma HLS INTERFACE axis port=cmd_to_cclo
#pragma HLS INTERFACE axis port=sts_from_cclo
#pragma HLS INTERFACE axis port=data_to_cclo
#pragma HLS INTERFACE axis port=data_from_cclo
#pragma HLS INTERFACE s_axilite port=return

    STREAM<int> mem_wr_done;
#pragma HLS stream variable=mem_wr_done depth=4

    // Barrier
    accl_hls::barrier_root(
        local_rank,
        comm_size,
        comm_adr, 
        dpcfg_adr,
        cmd_to_cclo,
        sts_from_cclo,
        data_to_cclo,
        data_from_cclo
    );


    dlrm_agg_compute(
        dst,
        data_to_cclo,
        data_from_cclo,
        mem_wr_done
    );

    check_done(
        mem_wr_done,
        comm_adr, 
        dpcfg_adr,
        cmd_to_cclo,
        sts_from_cclo
    );


}

// void dlrm_agg(
//     int *dst,
//     int count,
//     // //parameters pertaining to CCLO config
//     // ap_uint<32> comm_adr, 
//     // ap_uint<32> dpcfg_adr,
//     // //streams to and from CCLO
//     // STREAM<command_word> &cmd_to_cclo,
//     // STREAM<command_word> &sts_from_cclo,
//     STREAM<stream_word> &data_to_cclo,
//     STREAM<stream_word> &data_from_cclo
// ){
// #pragma HLS INTERFACE s_axilite port=count
// // #pragma HLS INTERFACE s_axilite port=comm_adr
// // #pragma HLS INTERFACE s_axilite port=dpcfg_adr
// #pragma HLS INTERFACE m_axi port=dst offset=slave
// // #pragma HLS INTERFACE axis port=cmd_to_cclo
// // #pragma HLS INTERFACE axis port=sts_from_cclo
// #pragma HLS INTERFACE axis port=data_to_cclo
// #pragma HLS INTERFACE axis port=data_from_cclo
// #pragma HLS INTERFACE s_axilite port=return

// #pragma HLS dataflow disable_start_propagation

//     //set up interfaces
//     // accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);
//     accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
//     //pull data from CCLO and write it to dst
//     data.pull_to_mem(dst, count);
//     data.tie_off_krnl2cclo();
//     #ifndef ACCL_SYNTHESIS
//         std::cout << "dlrm_agg: finish" << "\n";
//     #endif
// }
