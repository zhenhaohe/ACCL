/*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
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
#pragma once

#include "ap_int.h"
#include "ap_utils.h"
#include "ap_axi_sdata.h"

#define DATA_WIDTH 512
#define DEST_WIDTH 8

typedef ap_axiu<DATA_WIDTH, 0, 0, DEST_WIDTH> stream_word;
typedef ap_axiu<32, 0, 0, 0> command_word;

//this is a work-around for hlslib streams not synthesizing
//with vitis hls. Instead, we test a macro definition to select
//between simulation behaviour (use hlslib streams) and
//synthesis behaviour (use hls streams)
//all code using these macros should make sure it doesnt use features
//that aren't supported by both types of streams. For example,
//hlslib stream depths and storage types can't be defined on declaration
#ifdef ACCL_SYNTHESIS
//use hls streams
#include "hls_stream.h"
#define STREAM hls::stream 
#define STREAM_IS_EMPTY(s) s.empty()
#define STREAM_IS_FULL(s) s.full()
#define STREAM_READ(s) s.read()
#define STREAM_WRITE(s, val) s.write(val)
#else
//use hlslib streams
#include "Stream.h"
// #include <sstream>
// #include <iostream>
#define STREAM hlslib::Stream 
#define STREAM_IS_EMPTY(s) s.IsEmpty()
#define STREAM_IS_FULL(s) s.IsFull()
#define STREAM_READ(s) s.Pop()
#define STREAM_WRITE(s, val) s.Push(val)
#endif

namespace accl_hls {

//Primitives
#define ACCL_COPY           1
#define ACCL_COMBINE        2
#define ACCL_SEND           3 
#define ACCL_RECV           4
//Collectives
#define ACCL_BCAST          5
#define ACCL_SCATTER        6
#define ACCL_GATHER         7
#define ACCL_REDUCE         8
#define ACCL_ALLGATHER      9
#define ACCL_ALLREDUCE      10
#define ACCL_REDUCE_SCATTER 11
#define ACCL_BARRIER        12
#define ACCL_ALLTOALL       13
#define ACCL_REDUCE_PUT     14
#define ACCL_NOP            255

// Raw interfaces to start the function call
inline void start(
    ap_uint<32> scenario,
    ap_uint<32> len,
    ap_uint<32> comm,
    ap_uint<32> root_src_dst,
    ap_uint<32> function,
    ap_uint<32> msg_tag,
    ap_uint<32> datapath_cfg,
    ap_uint<32> compression_flags,
    ap_uint<32> stream_flags,
    ap_uint<64> addra,
    ap_uint<64> addrb,
    ap_uint<64> addrc,
    STREAM<command_word > &cmd
){
#pragma HLS PIPELINE II=15
    command_word tmp;
    tmp.keep = 0xf;

    tmp.data=scenario; tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=len; tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=comm; tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=root_src_dst; tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=function; tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=msg_tag; tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=datapath_cfg; tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=compression_flags; tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=stream_flags; tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=addra(31,0); tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=addra(63,32); tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=addrb(31,0); tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=addrb(63,32); tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=addrc(31,0); tmp.last=0;
    STREAM_WRITE(cmd, tmp);
    tmp.data=addrc(63,32); tmp.last=1;
    STREAM_WRITE(cmd, tmp);
}

// Raw interface to consume the ack
inline void finalize(STREAM<command_word > &sts)
{
#pragma HLS PIPELINE II=1
    STREAM_READ(sts);
}

/**
 * @brief Class encapsulating ACCL command streams
 * 
 */
class ACCLCommand{
    public:
        /**
         * @brief Construct a new ACCLCommand object
         * 
         * @param cmd Reference to command stream to CCLO
         * @param sts Reference to status stream to CCLO
         * @param comm_adr Communicator ID
         * @param dpcfg_adr Address of datapath configuration in CCLO memory
         * @param cflags Compression flags
         * @param sflags Stream flags
         */
        ACCLCommand(STREAM<command_word > &cmd, STREAM<command_word > &sts,
                    ap_uint<32> comm_adr, ap_uint<32> dpcfg_adr,
                    ap_uint<32> cflags, ap_uint<32> sflags) : 
                    cmd(cmd), sts(sts), comm_adr(comm_adr), dpcfg_adr(dpcfg_adr), cflags(cflags), sflags(sflags) {}

        /**
         * @brief Construct a new ACCLCommand object
         * 
         * @param cmd Reference to command stream to CCLO
         * @param sts Reference to status stream to CCLO
         */
        ACCLCommand(STREAM<command_word > &cmd, STREAM<command_word > &sts) : 
                    ACCLCommand(cmd, sts, 0, 0, 0, 0) {}

    protected:
        STREAM<command_word > &cmd;
        STREAM<command_word > &sts;
        ap_uint<32> comm_adr;
        ap_uint<32> dpcfg_adr;
        ap_uint<32> cflags;
        ap_uint<32> sflags;

    public:

        /**
         * @brief Launch an ACCL call
         * 
         * @param scenario Indicates type of call (see defines)
         * @param len Length of buffers involved in call, in elements (not bytes)
         * @param comm ID of communicator
         * @param root_src_dst Either root, source or destination rank, depending on scenario
         * @param function Function ID for reduction-type scenarios
         * @param msg_tag Message tag
         * @param datapath_cfg Address of datapath configuration structure
         * @param compression_flags Compression flags
         * @param stream_flags Stream flags
         * @param addra Address of first operand, or zero if not in use
         * @param addrb Address of second operand, or zero if not in use
         * @param addrc Address of result, or zero if not in use
         */
        void start_call(
            ap_uint<32> scenario,
            ap_uint<32> len,
            ap_uint<32> comm,
            ap_uint<32> root_src_dst,
            ap_uint<32> function,
            ap_uint<32> msg_tag,
            ap_uint<32> datapath_cfg,
            ap_uint<32> compression_flags,
            ap_uint<32> stream_flags,
            ap_uint<64> addra,
            ap_uint<64> addrb,
            ap_uint<64> addrc
        ){
            command_word tmp;
            tmp.keep = 0xf;
            io_section:{
                #pragma HLS protocol fixed
                tmp.data=scenario; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=len; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=comm; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=root_src_dst; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=function; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=msg_tag; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=datapath_cfg; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=compression_flags; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=stream_flags; tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addra(31,0); tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addra(63,32); tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addrb(31,0); tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addrb(63,32); tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addrc(31,0); tmp.last=0;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
                tmp.data=addrc(63,32); tmp.last=1;
                STREAM_WRITE(cmd, tmp);
                ap_wait();
            }  
        }

        /**
         * @brief Wait for a previously-launched call to finish
         * 
         */
        void finalize_call(){
            STREAM_READ(sts);
        }

        /**
         * @brief Perform ACCL NOP
         */
        void nop(){
            start_call(
                ACCL_NOP, 0, 0, 0, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                0, 0, 0
            );
            finalize_call();
        }

        /**
         * @brief Perform ACCL local array copy
         * 
         * @param len Number of array elements
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void copy(  ap_uint<32> len,
                    ap_uint<64> src_addr,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_COPY, len, 0, 0, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief Perform ACCL local array combine
         * 
         * @param len Number of array elements
         * @param op0_addr Address of first operand array
         * @param op1_addr Address of second operand array
         * @param res_addr Address of result array
         */
        void combine(   ap_uint<32> len,
                        ap_uint<64> op0_addr,
                        ap_uint<64> op1_addr,
                        ap_uint<64> res_addr
        ){
            start_call(
                ACCL_COMBINE, len, 0, 0, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                op0_addr, op1_addr, res_addr
            );
            finalize_call();
        }

        /**
         * @brief Send data to a remote peer. Two-sided, i.e. requires a recv on remote end.
         * 
         * @param len Number of array elements
         * @param tag Message tag
         * @param dst_rank Rank ID of destination
         * @param src_addr Source array address
         */
        void send(  ap_uint<32> len,
                    ap_uint<32> tag,
                    ap_uint<32> dst_rank,
                    ap_uint<64> src_addr
        ){
            start_call(
                ACCL_SEND, len, comm_adr, dst_rank, 0, tag, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, 0
            );
            finalize_call();
        }

        /**
         * @brief One-sided data transfer to a stream on a remote peer. 
         * 
         * @param len Number of array elements
         * @param stream_id Stream ID at destination. IDs >=247 are reserved, call will not execute if set in this range
         * @param dst_rank Rank ID of destination
         * @param src_addr Source array address
         */
        void stream_put(ap_uint<32> len,
                        ap_uint<32> stream_id,
                        ap_uint<32> dst_rank,
                        ap_uint<64> src_addr
        ){
            if(stream_id > 246) return;
            start_call(
                ACCL_SEND, len, comm_adr, dst_rank, 0, stream_id, 
                dpcfg_adr, cflags, sflags | 0x2, 
                src_addr, 0, 0
            );
            finalize_call();
        }

        /**
         * @brief One-sided data transfer to a stream on a remote peer. 
         * 
         * @param len Number of array elements
         * @param stream_id Stream ID at destination. IDs >=247 are reserved, call will not execute if set in this range
         * @param dst_rank Rank ID of destination
         * @param src_addr Source array address
         */
        void stream_put_nb(ap_uint<32> len,
                        ap_uint<32> stream_id,
                        ap_uint<32> dst_rank,
                        ap_uint<64> src_addr
        ){
            if(stream_id > 246) return;
            start_call(
                ACCL_SEND, len, comm_adr, dst_rank, 0, stream_id, 
                dpcfg_adr, cflags, sflags | 0x2, 
                src_addr, 0, 0
            );
        }

        /**
         * @brief Receive data send from a remote peer.
         * 
         * @param len Number of array elements
         * @param tag Message tag
         * @param src_rank Rank ID of sender
         * @param dst_addr Destination array address
         */
        void recv(  ap_uint<32> len,
                    ap_uint<32> tag,
                    ap_uint<32> src_rank,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_RECV, len, comm_adr, src_rank, 0, tag, 
                dpcfg_adr, cflags, sflags,
                0, dst_addr, 0
            );
            finalize_call();
        }

        /**
         * @brief Broadcast data to members of the communicator
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param src_addr Source array address
         */
        void bcast( ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<64> src_addr
        ){
            start_call(
                ACCL_BCAST, len, comm_adr, root, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, 0
            );
            finalize_call();
        }

        /**
         * @brief Scatter data to members of the communicator
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void scatter( ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<64> src_addr,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_SCATTER, len, comm_adr, root, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief Gather data from members of the communicator
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void gather(ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<64> src_addr,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_GATHER, len, comm_adr, root, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief All-gather data in the communicator. Equivalent to gather followed by broadcast
         * 
         * @param len Number of array elements
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void all_gather(ap_uint<32> len,
                        ap_uint<64> src_addr,
                        ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_ALLGATHER, len, comm_adr, 0, 0, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief Reduce data memory to memory
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param function Reduction function ID
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void reduce(ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<32> function,
                    ap_uint<64> src_addr,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_REDUCE, len, comm_adr, root, function, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief Reduce data stream to memory
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param function Reduction function ID
         * @param dst_addr Destination array address
         */
        void reduce(ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<32> function,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_REDUCE, len, comm_adr, root, function, 0, 
                dpcfg_adr, cflags, 1, 
                0, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief Reduce data stream to stream
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param function Reduction function ID
         */
        void reduce(ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<32> function
        ){
            start_call(
                ACCL_REDUCE, len, comm_adr, root, function, 0, 
                dpcfg_adr, cflags, 3, 
                0, 0, 0
            );
            finalize_call();
        }

        /**
         * @brief Reduce data stream to stream, exclusive of root,
         *        with final delivery to root via stream put
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param function Reduction function ID
         */
        void reduce_put(ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<32> function
        ){
            start_call(
                ACCL_REDUCE_PUT, len, comm_adr, root, function, 0, 
                dpcfg_adr, cflags, 3, 
                0, 0, 0
            );
            finalize_call();
        }

        /**
         * @brief Reduce data stream to stream, exclusive of root,
         *        with final delivery to root via stream put
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param function Reduction function ID
         */
        void reduce_put_nb(ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<32> function
        ){
            start_call(
                ACCL_REDUCE_PUT, len, comm_adr, root, function, 0, 
                dpcfg_adr, cflags, 3, 
                0, 0, 0
            );
        }


        /**
         * @brief Reduce data stream to stream
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param function Reduction function ID
         */
        void reduce_nb(ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<32> function
        ){
            start_call(
                ACCL_REDUCE, len, comm_adr, root, function, 0, 
                dpcfg_adr, cflags, 3, 
                0, 0, 0
            );
        }

        /**
         * @brief Reduce data memory to memory
         * 
         * @param len Number of array elements
         * @param root Rank ID of root node
         * @param function Reduction function ID
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void reduce_nb(ap_uint<32> len,
                    ap_uint<32> root,
                    ap_uint<32> function,
                    ap_uint<64> src_addr,
                    ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_REDUCE, len, comm_adr, root, function, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
        }

        /**
         * @brief Reduce-scatter data in the communicator. Equivalent to reduce followed by scatter
         * 
         * @param len Number of array elements
         * @param function Reduction function ID
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void reduce_scatter(ap_uint<32> len,
                            ap_uint<32> function,
                            ap_uint<64> src_addr,
                            ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_REDUCE_SCATTER, len, comm_adr, 0, function, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }

        /**
         * @brief All-reduce data in the communicator. Equivalent to reduce followed by broadcast
         * 
         * @param len Number of array elements
         * @param function Reduction function ID
         * @param src_addr Source array address
         * @param dst_addr Destination array address
         */
        void all_reduce(ap_uint<32> len,
                        ap_uint<32> function,
                        ap_uint<64> src_addr,
                        ap_uint<64> dst_addr
        ){
            start_call(
                ACCL_ALLREDUCE, len, comm_adr, 0, function, 0, 
                dpcfg_adr, cflags, sflags, 
                src_addr, 0, dst_addr
            );
            finalize_call();
        }
};

/**
 * @brief Class encapsulating ACCL data streams
 * 
 */
class ACCLData{
    public:
        /**
         * @brief Construct a new ACCLData object
         * 
         * @param krnl2cclo Reference to data stream from user kernel to CCLO
         * @param cclo2krnl Reference to data stream from CCLO to user kernel
         */
        ACCLData(STREAM<stream_word> &krnl2cclo, STREAM<stream_word> &cclo2krnl) : 
                    cclo2krnl(cclo2krnl), krnl2cclo(krnl2cclo){}

    protected:
        STREAM<stream_word> &krnl2cclo;
        STREAM<stream_word> &cclo2krnl;

    public:
        /**
         * @brief Push user data to the CCLO
         * 
         * @param data Data word (64B)
         * @param dest Destination value (potentially used in routing)
         */
        void push(ap_uint<DATA_WIDTH> data, ap_uint<DEST_WIDTH> dest){
            stream_word tmp;
            tmp.data = data;
            tmp.dest = dest;
            tmp.last = 1;
            tmp.keep = -1;
            krnl2cclo.write(tmp);
        }

        /**
         * @brief Pull data from CCLO stream
         * 
         * @return stream_word
         */
        stream_word pull(){
            return cclo2krnl.read();   
        }

        void tie_off_krnl2cclo()
        {
            STREAM<ap_uint<512> > tmp_stream;
            if (!tmp_stream.empty())
            {
                ap_uint<512> tmpword = tmp_stream.read();
                stream_word tmp;
                tmp.data = tmpword;
                tmp.dest = 0;
                tmp.last = 1;
                tmp.keep = -1;
                krnl2cclo.write(tmp);
            }
        }

        void tie_off_cclo2krnl()
        {
            if (!cclo2krnl.empty())
            {
                stream_word tmpword = cclo2krnl.read();   
            }
        }

        /**
         * @brief Read data from memory and push user data to the CCLO
         * 
         * @param data Data word (64B)
         * @param dest Destination value (potentially used in routing)
         */
        void push_from_mem (int* src, int count, ap_uint<DEST_WIDTH> dest=0)
        {
            ap_uint<512> tmpword;
            int word_count = 0;
            int rd_count = count;
            while(rd_count > 0){
                // #ifndef ACCL_SYNTHESIS
                //     std::cout << "push_from_mem"<<"\n";
                // #endif
                //read 16 ints into a 512b vector
                for(int i=0; (i<16) && (rd_count>0); i++){
                    int inc = src[i+16*word_count];
                    tmpword((i+1)*32-1,i*32) = *reinterpret_cast<ap_uint<32>*>(&inc);
                    rd_count--;
                    // #ifndef ACCL_SYNTHESIS
                    //     std::cout <<"src:"<<inc<<" word:"<<tmpword((i+1)*32-1,i*32)<<std::endl;
                    // #endif
                }
                //send the vector to cclo
                push(tmpword, dest);
                word_count++;
            }
        }

        /**
         * @brief Read data from stream and push user data to the CCLO
         * 
         * @param data Data word (64B)
         * @param dest Destination value (potentially used in routing)
         */
        void push_from_stream (STREAM<ap_uint<512> > &src_stream, int count, ap_uint<DEST_WIDTH> dest=0)
        {
            ap_uint<512> tmpword;
            int rd_count = (count + 15) / 16;
            while(rd_count > 0){
                tmpword = src_stream.read();
                //send the vector to cclo
                push(tmpword, dest);
                rd_count--;
            }
            #ifndef ACCL_SYNTHESIS
                std::cout <<"push_from_stream finish rd_count:"<<rd_count<<std::endl;
            #endif
            
        }

        /**
         * @brief Pull data from CCLO stream and store it in mem
         * 
         * @return stream_word
         */
        void pull_to_mem (int* dst, int count)
        {
            ap_uint<512> tmpword;
            int word_count = 0;
            int wr_count = count;
            word_count = 0;
            // #ifndef ACCL_SYNTHESIS
            //     std::cout << "pull_to_mem"<<"\n";
            // #endif
            while(wr_count > 0){
                //read vector from CCLO
                tmpword = pull().data;
                //read from the 512b vector into 16 ints
                for(int i=0; (i<16) && (wr_count>0); i++){
                    ap_uint<32> val = tmpword((i+1)*32-1,i*32);
                    dst[i+16*word_count] = *reinterpret_cast<int*>(&val);
                    wr_count--;
                    // #ifndef ACCL_SYNTHESIS
                    //     std::cout <<"word:"<<val<<" dst:"<<dst[i+16*word_count]<<std::endl;
                    // #endif
                }
                word_count++;
            }
        }

        /**
         * @brief Pull data from CCLO stream and store it to stream
         * 
         * @return stream_word
         */
        void pull_to_stream (STREAM<ap_uint<512> > &dst_stream, int count)
        {
            ap_uint<512> tmpword;
            int wr_count = (count + 15) / 16 ;
            while(wr_count > 0){
                //read vector from CCLO
                tmpword = pull().data;
                dst_stream.write(tmpword);
                wr_count--;
            }
            #ifndef ACCL_SYNTHESIS
                std::cout <<"pull_to_stream finish: wr_count:"<<wr_count<<std::endl;
            #endif
        }
};

#ifndef ACCL_SYNTHESIS
// Pass simulation but doesn't pass the hardware execution
inline void barrier_root(
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
)
{
    //set up interfaces
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);

    // Barrier
    // all non-root nodes send 1 packet to root
    // root receives the packets and then send one packet to each non-root node
    ap_uint<512> tmpword = 0;
    ap_uint<32> root = comm_size-1;
    
    collect_msg:
    for (int i = 0; i < (comm_size-1); i++)
    {
    #pragma HLS PIPELINE II=1
        tmpword = data.pull().data;
    }
    #ifndef ACCL_SYNTHESIS
        std::cout << "barrier root receive messages:" <<comm_size-1<< "\n";
    #endif
    // response to non-root ranks
    response_msg:
    for (int i = 0; i < (comm_size-1); i++)
    {
        accl_hls::start(ACCL_SEND, 16, comm_adr, i, 0, 9, dpcfg_adr, 0, 3, 0, 0, 0, cmd_to_cclo);
        #ifndef ACCL_SYNTHESIS
            std::cout << "barrier root stream put command:"<<i<< "\n";
        #endif
        data.push(tmpword, 0);
        #ifndef ACCL_SYNTHESIS
            std::cout << "barrier root stream put data:"<<i<< "\n";
        #endif
        accl_hls::finalize(sts_from_cclo);
        #ifndef ACCL_SYNTHESIS
            std::cout << "barrier root stream put ack:"<<i<< "\n";
        #endif
    }

    #ifndef ACCL_SYNTHESIS
        std::cout << "barrier: finish" << "\n";
    #endif
    
}

inline void barrier_non_root(
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
)
{
    //set up interfaces
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);

    // Barrier
    // all non-root nodes send 1 packet to root
    // root receives the packets and then send one packet to each non-root node
    ap_uint<512> tmpword = 0;
    ap_uint<32> root = comm_size-1;

    // send one packet to root
    accl_hls::start(ACCL_SEND, 16, comm_adr, root, 0, 9, dpcfg_adr, 0, 3, 0, 0, 0, cmd_to_cclo);

    data.push(tmpword, 0);

    accl_hls::finalize(sts_from_cclo);

    #ifndef ACCL_SYNTHESIS
        std::cout << "barrier non root stream put one message"<< "\n";
    #endif

    // wait response from root
    tmpword = data.pull().data;
    
    #ifndef ACCL_SYNTHESIS
        std::cout << "barrier non root receive one message"<< "\n";
    #endif
    
    #ifndef ACCL_SYNTHESIS
        std::cout << "barrier: finish" << "\n";
    #endif
}
#else 
// pass hardware execution but doesn't pass simulation
// Key is to have a seperate loop for starting the call command
inline void barrier_root(
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
)
{
    //set up interfaces
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);

    // Barrier
    // all non-root nodes send 1 packet to root
    // root receives the packets and then send one packet to each non-root node
    unsigned int root = comm_size-1;
    
    collect_msg:
    for (unsigned int i = 0; i < root; i++)
    {
    #pragma HLS PIPELINE II=1
        ap_uint<512> tmpword = data.pull().data;
    }
    #ifndef ACCL_SYNTHESIS
        std::cout << "barrier root receive messages:" <<comm_size-1<< "\n";
    #endif
    // response to non-root ranks
    response_command:
    for (unsigned int i = 0; i < root; i++)
    {
        accl.stream_put_nb(16, 9, i, 0); 
        #ifndef ACCL_SYNTHESIS
            std::cout << "barrier root stream put command:"<<i<< "\n";
        #endif
    }
    response_data:
    for (unsigned int i = 0; i < root; i++)
    {
        #pragma HLS PIPELINE II=1
        ap_uint<512> tmpword = 1;
        data.push(tmpword, 0);
        #ifndef ACCL_SYNTHESIS
            std::cout << "barrier root stream put data:"<<i<< "\n";
        #endif
    }
    response_ack:
    for (unsigned int i = 0; i < root; i++)
    {
        accl.finalize_call();
        #ifndef ACCL_SYNTHESIS
            std::cout << "barrier root stream put ack:"<<i<< "\n";
        #endif
    }

    #ifndef ACCL_SYNTHESIS
        std::cout << "barrier: finish" << "\n";
    #endif
    
}

inline void barrier_non_root(
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
)
{
    //set up interfaces
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);

    // Barrier
    // all non-root nodes send 1 packet to root
    // root receives the packets and then send one packet to each non-root node
    ap_uint<512> tmpword = 1;
    ap_uint<32> root = comm_size-1;

    // send one packet to root
    accl.stream_put_nb(16, 9, root, 0); 

    data.push(tmpword, 0);

    accl.finalize_call();

    #ifndef ACCL_SYNTHESIS
        std::cout << "barrier non root stream put one message"<< "\n";
    #endif

    // wait response from root
    tmpword = data.pull().data;
    
    #ifndef ACCL_SYNTHESIS
        std::cout << "barrier non root receive one message"<< "\n";
    #endif
    
    #ifndef ACCL_SYNTHESIS
        std::cout << "barrier: finish" << "\n";
    #endif
}
#endif

}


