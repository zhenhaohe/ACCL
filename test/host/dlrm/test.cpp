/*******************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
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

#include "accl.hpp"
#include <cstdlib>
#include <functional>
#include <mpi.h>
#include <random>
#include <sstream>
#include <tclap/CmdLine.h>
#include <vector>
#include "vadd_put.h"
#include "cclo_bfm.h"
#include <xrt/xrt_device.h>
#include <iostream>
#include <fstream>
#include "dlrm.h"
#include "constants.hpp"

using namespace ACCL;

//******************************
//**  XCC Operations          **
//******************************
//Housekeeping
#define ACCL_CONFIG         0
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
#define ACCL_NOP            255

//ACCL_CONFIG SUBFUNCTIONS
#define HOUSEKEEP_SWRST                0
#define HOUSEKEEP_PKTEN                1
#define HOUSEKEEP_TIMEOUT              2
#define HOUSEKEEP_OPEN_PORT            3
#define HOUSEKEEP_OPEN_CON             4
#define HOUSEKEEP_SET_STACK_TYPE       5
#define HOUSEKEEP_SET_MAX_SEGMENT_SIZE 6
#define HOUSEKEEP_CLOSE_CON            7

#define CONFIG_HW_BENCH_RECORD 7

int rank, size;
unsigned failed_tests;
unsigned skipped_tests;

struct options_t {
  int start_port;
  unsigned int rxbuf_size;
  unsigned int seg_size;
  unsigned int count;
  unsigned int nruns;
  unsigned int device_index;
  unsigned int num_rxbufmem;
  unsigned int test_mode;
  bool test_xrt_simulator;
  bool debug;
  bool hardware;
  bool axis3;
  bool udp;
	bool tcp;
  bool hw_bench;
  bool enableUserKernel;
  std::string xclbin;
	std::string fpgaIP;
};

struct timestamp_t {
  uint64_t cmdSeq;
  uint64_t scenario;
  uint64_t len;
  uint64_t comm;
  uint64_t root_src_dst;
  uint64_t function;
  uint64_t msg_tag;
  uint64_t datapath_cfg;
  uint64_t compression_flags;
  uint64_t stream_flags;
  uint64_t addra_l;
  uint64_t addra_h;
  uint64_t addrb_l;
  uint64_t addrb_h;
  uint64_t addrc_l;
  uint64_t addrc_h;
  uint64_t cmdTimestamp;
  uint64_t cmdEnd;
  uint64_t stsSeq;
  uint64_t sts;
  uint64_t stsTimestamp;
  uint64_t stsEnd;
};

timestamp_t readTimeStamp(uint64_t* host_ptr_hw_bench_cmd, uint64_t* host_ptr_hw_bench_sts, unsigned int& cmd_mem_offset, unsigned int& sts_mem_offset)
{
  timestamp_t timestamp;

  timestamp.cmdSeq = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.scenario = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.len = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.comm = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.root_src_dst = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.function = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.msg_tag = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.datapath_cfg = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.compression_flags = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.stream_flags = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addra_l = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addra_h = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addrb_l = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addrb_h = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addrc_l = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.addrc_h = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.cmdTimestamp = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;
  timestamp.cmdEnd = (uint64_t)host_ptr_hw_bench_cmd[cmd_mem_offset];
  cmd_mem_offset++;

  timestamp.stsSeq = (uint64_t)host_ptr_hw_bench_sts[sts_mem_offset];
  sts_mem_offset++;
  timestamp.sts = (uint64_t)host_ptr_hw_bench_sts[sts_mem_offset];
  sts_mem_offset++;
  timestamp.stsTimestamp = (uint64_t)host_ptr_hw_bench_sts[sts_mem_offset];
  sts_mem_offset++;
  timestamp.stsEnd = (uint64_t)host_ptr_hw_bench_sts[sts_mem_offset];
  sts_mem_offset++;

  return timestamp;
}

void printTimeStamp(timestamp_t timestamp, options_t &options)
{
  std::string exp;
  bool writeToLog = false;
  std::cout<<"cmdSeq: "<<timestamp.cmdSeq<<" ";
  switch (timestamp.scenario)
  {
      case ACCL_COPY:
          std::cout<<"ACCL_COPY";
          break;
      case ACCL_COMBINE:
          std::cout<<"ACCL_COMBINE";
          break;
      case ACCL_SEND:
          std::cout<<"ACCL_SEND";
          exp="sendrecv_K2K";
          writeToLog=true;
          break;
      case ACCL_RECV:
          std::cout<<"ACCL_RECV";
          exp="sendrecv_K2K";
          writeToLog=true;
          break;
      case ACCL_BCAST:
          std::cout<<"ACCL_BCAST";
          exp="bcast_K2K";
          writeToLog=true;
          break;
      case ACCL_SCATTER:
          std::cout<<"ACCL_SCATTER";
          exp="scatter_K2K";
          writeToLog=true;
          break;
      case ACCL_GATHER:
          std::cout<<"ACCL_GATHER";
          exp="gather_K2K";
          writeToLog=true;
          break;
      case ACCL_REDUCE:
          std::cout<<"ACCL_REDUCE";
          exp="reduce_K2K";
          writeToLog=true;
          break;
      case ACCL_ALLGATHER:
          std::cout<<"ACCL_ALLGATHER";
          exp="allgather_K2K";
          writeToLog=true;
          break;
      case ACCL_REDUCE_SCATTER:
          std::cout<<"ACCL_REDUCE_SCATTER";
          break;
      case ACCL_ALLREDUCE:
          std::cout<<"ACCL_ALLREDUCE";
          exp="allreduce_K2K";
          writeToLog=true;
          break;
      case ACCL_BARRIER:
          std::cout<<"ACCL_COPY";
          break;
      case ACCL_ALLTOALL:
          std::cout<<"ACCL_ALLTOALL";
          break;
      case ACCL_NOP:
          std::cout<<"ACCL_NOP";
          break;
      case ACCL_CONFIG:
          std::cout<<"ACCL_CONFIG";
          switch (timestamp.function)
          {
              case HOUSEKEEP_SWRST:
                  std::cout<<" HOUSEKEEP_SWRST";
                  break;
              case HOUSEKEEP_PKTEN:
                  std::cout<<" HOUSEKEEP_PKTEN";
                  break;
              case HOUSEKEEP_TIMEOUT:
                  std::cout<<" HOUSEKEEP_TIMEOUT";
                  break;
              case HOUSEKEEP_OPEN_PORT:
                  std::cout<<" HOUSEKEEP_OPEN_PORT";
                  break;
              case HOUSEKEEP_OPEN_CON:
                  std::cout<<" HOUSEKEEP_OPEN_CON";
                  break;
              case HOUSEKEEP_CLOSE_CON:
                  std::cout<<" HOUSEKEEP_CLOSE_CON";
                  break;
              case HOUSEKEEP_SET_STACK_TYPE:
                  std::cout<<" HOUSEKEEP_SET_STACK_TYPE";
                  break;
              case HOUSEKEEP_SET_MAX_SEGMENT_SIZE:
                  std::cout<<" HOUSEKEEP_SET_MAX_SEGMENT_SIZE";
                  break;
              default:
                  std::cout<<" Not Recognized Function:"<<timestamp.function;
                  break;
          }
          break;
      default:
          std::cout<<"Not Recognized Scenario:"<<timestamp.scenario;
          writeToLog=false;
          break;
  }
  std::cout<<" len: "<<timestamp.len<<" ";
  std::cout<<" cmdTimestamp: "<<timestamp.cmdTimestamp<<" ";
  std::cout<<" cmdEnd: "<<timestamp.cmdEnd<<" ";
  std::cout<<" sts: "<<timestamp.sts<<" ";
  std::cout<<" stsTimestamp: "<<timestamp.stsTimestamp<<" ";
  std::cout<<" stsEnd: "<<timestamp.stsEnd<<" ";
  std::cout<<std::endl;

  if ((timestamp.cmdEnd != 0xFFFFFFFFFFFFFFFF) || (timestamp.stsEnd != 0xFFFFFFFFFFFFFFFF))
  {
    writeToLog=false;
  }
  // if (writeToLog)
  // {
  //   uint64_t start_cycle = timestamp.cmdTimestamp;
  //   uint64_t end_cycle = timestamp.stsTimestamp;
  //   double durationUs = (end_cycle-start_cycle)/(double)FREQ;
  //   double tput = (options.count*sizeof(float)*8.0)/(durationUs*1000.0); // only useful for send/recv
  //   accl_log(rank, format_log(exp, options, durationUs, tput));
  // }
}


std::string prepend_process() {
  return "[process " + std::to_string(rank) + "] ";
}

void test_debug(std::string message, options_t &options) {
  if (options.debug) {
    std::cerr << message << std::endl;
  }
}

std::unique_ptr<ACCL::ACCL> test_vadd_put(options_t options) {
    std::vector<rank_t> ranks = {};
    for (int i = 0; i < size; ++i) {
        rank_t new_rank = {"127.0.0.1", options.start_port + i, i, options.rxbuf_size};
        ranks.emplace_back(new_rank);
    }

    std::unique_ptr<ACCL::ACCL> accl;
    xrt::device device;

    if (options.hardware) {
        device = xrt::device(options.device_index);
        auto xclbin_uuid = device.load_xclbin(options.xclbin);
        auto cclo_ip = xrt::ip(device, device.get_xclbin_uuid(),
                            "ccl_offload:{ccl_offload_" + std::to_string(rank) + "}");
        auto hostctrl_ip =
            xrt::kernel(device, device.get_xclbin_uuid(), "hostctrl:{hostctrl_" + std::to_string(rank) + "_0}",
                        xrt::kernel::cu_access_mode::exclusive);

        int devicemem = rank * 6;
        std::vector<int> rxbufmem = {rank * 6 + 1};
        int networkmem = rank * 6 + 2;

        accl = std::make_unique<ACCL::ACCL>(
            ranks, rank, device, cclo_ip, hostctrl_ip, devicemem, rxbufmem,
            networkProtocol::UDP, options.num_rxbufmem, options.rxbuf_size);
    } else {
        accl = std::make_unique<ACCL::ACCL>(ranks, rank, options.start_port,
                                                options.udp ? networkProtocol::UDP : networkProtocol::TCP, options.num_rxbufmem,
                                                options.rxbuf_size);
    }

    accl->set_timeout(1e8);
    std::cout << "Host-side CCLO initialization finished" << std::endl;

    // barrier here to make sure all the devices are configured before testing
    MPI_Barrier(MPI_COMM_WORLD);

    //run test here:

    //allocate float arrays for the HLS function to use
    float src[options.count], dst[options.count];
    for(int i=0; i<options.count; i++){
        src[i] = 1.0*(options.count*rank+i);
    }

    if (options.hardware) {
        auto vadd_ip = xrt::kernel(device, device.get_xclbin_uuid(), "vadd_put:{vadd_" + std::to_string(rank) + "_0}",
                        xrt::kernel::cu_access_mode::exclusive);
        //need to use XRT API because vadd kernel might use different HBM banks than ACCL
        auto src_bo = xrt::bo(device, sizeof(float)*options.count, vadd_ip.group_id(0));
        auto dst_bo = xrt::bo(device, sizeof(float)*options.count, vadd_ip.group_id(1));

        src_bo.write(src);
        src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = vadd_ip(src_bo, dst_bo, options.count, (rank+1)%size, accl->get_communicator_addr(),
                    accl->get_arithmetic_config_addr({dataType::float32, dataType::float32}));
        run.wait(10000);

        dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        dst_bo.read(dst);
    } else {
        //initialize a CCLO BFM and streams as needed
        hlslib::Stream<command_word> callreq, callack;
        hlslib::Stream<stream_word> data_cclo2krnl, data_krnl2cclo;
        std::vector<unsigned int> dest = {9};
        CCLO_BFM cclo(options.start_port, rank, size, dest, callreq, callack, data_cclo2krnl, data_krnl2cclo);
        cclo.run();
        std::cout << "CCLO BFM started" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);

        //run the hls function, using the global communicator
        vadd_put(   src, dst, options.count,
                    (rank+1)%size,
                    accl->get_communicator_addr(),
                    accl->get_arithmetic_config_addr({dataType::float32, dataType::float32}),
                    callreq, callack,
                    data_krnl2cclo, data_cclo2krnl);
        //stop the BFM
        cclo.stop();
    }

    //check HLS function outputs
    unsigned int err_count = 0;
    for(int i=0; i<options.count; i++){
        float expected = 1.0*(options.count*((rank+size-1)%size)+i) + 1;
        if(dst[i] != expected){
            err_count++;
            std::cout << "Mismatch at [" << i << "]: got " << dst[i] << " vs expected " << expected << std::endl;
        }
    }

    std::cout << "Test finished with " << err_count << " errors" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    return accl;
}

template <typename T> static void random_array(T *data, size_t count) {
  std::uniform_real_distribution<T> distribution(-1000, 1000);
  std::mt19937 engine;
  auto generator = std::bind(distribution, engine);
  for (size_t i = 0; i < count; ++i) {
    data[i] = generator();
  }
}

template <typename T> std::unique_ptr<T> random_array(size_t count) {
  std::unique_ptr<T> data(new T[count]);
  random_array(data.get(), count);
  return data;
}

void test_allreduce(ACCL::ACCL &accl, options_t &options,
                    reduceFunction function) {
  std::cout << "Start allreduce test and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reducing data...", options);
  accl.allreduce(*op_buf, *res_buf, count, function);

  int errors = 0;

  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i] * size;

    if (res != ref) {
      // std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
      //                  std::to_string(res) + " != " + std::to_string(ref) + ")"
      //           << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }
}

void test_reduce_put(ACCL::ACCL &accl, options_t &options, int root,
                 reduceFunction function) {
  std::cout << "Start reduce-put test..." << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_buffer<float>(count, dataType::float32);
  auto res_buf = accl.create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  random_array(res_buf->buffer(), count);
  res_buf->sync_to_device();

  if (rank != root) {
    test_debug("Loading stream on rank" + std::to_string(rank) + "...", options);
    accl.copy_to_stream(*op_buf, count, false);
  }

  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl.reduce_put(dataType::float32, dataType::float32, count, root, function);

  if (rank == root) {
    int errors = 0;

    test_debug("Unloading stream on rank" + std::to_string(rank) + "...", options);
    accl.copy_from_stream(*res_buf, count, false);

    for (unsigned int i = 0; i < count; ++i) {
      float res = (*res_buf)[i];
      float ref = (*op_buf)[i] * (size-1);

      if (res != ref) {
        std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
                         std::to_string(res) + " != " + std::to_string(ref) +
                         ")"
                  << std::endl;
        errors += 1;
      }
    }

    if (errors > 0) {
      std::cout << std::to_string(errors) + " errors!" << std::endl;
      failed_tests++;
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  } else {
    test_debug("Done on rank" + std::to_string(rank), options);
  }
}

// Total node in the system: 2*numEmbedNodes+2
// Reduce node rank(0) to rank(numEmbedNodes) with root at rank(numEmbedNodes)
// Embedding node rank(numEmbedNodes+1) to rank(2*numEmbedNodes)
// Final aggregate node rank(2*numEmbedNodes+1)
// Global communicator contains rank(0) to rank(2*numEmbedNodes+1)
// Reduction communicator contains rank(0) to rank(numEmbedNodes)
void test_dlrm_sim(ACCL::ACCL &accl, options_t &options, unsigned int numEmbedNodes) {
  std::cout << "Start dlrm_kernels simulation test with " <<numEmbedNodes<<" embedding nodes..."<< std::endl;
  unsigned int count = options.count;
  unsigned int own_rank = rank;
  int errors = 0;
  if (size < 6) {
    std::cout<<"Error: Minimum group size is 6, current size:"<<size<<std::endl;
    return;
  }
	
	unsigned int reduce_comm_size = numEmbedNodes+1;
  unsigned int reduce_comm_root = numEmbedNodes;
	
	unsigned int own_role;
	if ((own_rank > numEmbedNodes) && (own_rank < 2*numEmbedNodes+1)){
		own_role = DLRM_EMBED_ROLE;
		std::cout<<"Rank:"<<own_rank<<" is DLRM_EMBED_ROLE"<<std::endl;
	} else if (own_rank<reduce_comm_size){
		if (own_rank == reduce_comm_root){
			own_role = DLRM_REDUCE_ROOT_ROLE;
			std::cout<<"Rank:"<<own_rank<<" is DLRM_REDUCE_ROOT_ROLE"<<std::endl;
		} else {
			own_role = DLRM_REDUCE_SLAVE_ROLE;
			std::cout<<"Rank:"<<own_rank<<" is DLRM_REDUCE_SLAVE_ROLE"<<std::endl;
		}
	} else if (own_rank == 2*numEmbedNodes+1){
		own_role = DLRM_AGG_ROLE;
		std::cout<<"Rank:"<<own_rank<<" is DLRM_AGG_ROLE"<<std::endl;
	} else{
		std::cout<<"Error: rank not assigned to a role!"<<std::endl;
    return;
	}
	  
  // create reduce group from rank 0 to rank numEmbedNodes
	auto group = accl.get_comm_group(GLOBAL_COMM);
  std::vector<rank_t> reduce_group;
  communicatorId REDUCE_COMM;
  if (own_role == DLRM_REDUCE_ROOT_ROLE || own_role == DLRM_REDUCE_SLAVE_ROLE)
  {
    for (unsigned int i = 0; i < reduce_comm_size; i++)
    {
      reduce_group.push_back(group[i]);
    }
    REDUCE_COMM = accl.create_communicator(reduce_group, own_rank);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  test_debug(accl.dump_communicator(), options);

	// create buffers
  int host_embed_buf[options.count], host_op_buf[options.count], host_res_buf[options.count];

  for (unsigned int i = 0; i < count; i++) {
    host_embed_buf[i] = (int)i;
  }
	
  for (unsigned int i = 0; i < count; i++) {
    host_op_buf[i] = 0;
  }

  for (unsigned int i = 0; i < count; i++) {
    host_res_buf[i] = 0;
  }


  std::cout<<"host_embed_buf:"<<host_embed_buf<<" host_op_buf:"<<host_op_buf<<" host_res_buf:"<<host_res_buf<<std::endl;

	//initialize a CCLO BFM and streams as needed
	hlslib::Stream<command_word> callreq, callack;
	hlslib::Stream<stream_word> data_cclo2krnl, data_krnl2cclo;
	std::vector<unsigned int> dest = {0,1,2,3,4,5,6,7,8,9}; 
	CCLO_BFM cclo(options.start_port, rank, size, dest, callreq, callack, data_cclo2krnl, data_krnl2cclo);
	cclo.run();
	std::cout << "CCLO BFM started" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  // Embedding nodes
  if (own_role == DLRM_EMBED_ROLE)
  {
    unsigned int dst_rank = own_rank - reduce_comm_size;
    dlrm_embedding(
            host_embed_buf, 
            host_embed_buf,
            host_embed_buf,
            host_embed_buf,
            host_embed_buf,
            host_embed_buf, 
            host_embed_buf,
            host_embed_buf,
            dst_rank, own_rank, size, accl.get_communicator_addr(GLOBAL_COMM), accl.get_arithmetic_config_addr({dataType::int32, dataType::int32}), callreq, callack, data_krnl2cclo, data_cclo2krnl);
  }
  // recv and reduce operation for reduce nodes
  else if (own_role == DLRM_REDUCE_ROOT_ROLE)
  {
    unsigned int destination = 2*numEmbedNodes+1;
    dlrm_reduce_root(destination, reduce_comm_root, (unsigned int)ACCL::reduceFunction::SUM, own_rank, size, accl.get_communicator_addr(GLOBAL_COMM), accl.get_communicator_addr(REDUCE_COMM), accl.get_arithmetic_config_addr({dataType::int32, dataType::int32}), callreq, callack, data_krnl2cclo, data_cclo2krnl);
  } else if (own_role == DLRM_REDUCE_SLAVE_ROLE)
  {
    dlrm_reduce_slave(reduce_comm_root, (unsigned int)ACCL::reduceFunction::SUM, own_rank, size, accl.get_communicator_addr(GLOBAL_COMM), accl.get_communicator_addr(REDUCE_COMM), accl.get_arithmetic_config_addr({dataType::int32, dataType::int32}), callreq, callack, data_krnl2cclo, data_cclo2krnl);
  } else if (own_role == DLRM_AGG_ROLE)
  {
    dlrm_agg(host_res_buf, own_rank, size, accl.get_communicator_addr(GLOBAL_COMM), accl.get_arithmetic_config_addr({dataType::int32, dataType::int32}), callreq, callack, data_krnl2cclo, data_cclo2krnl);
  }

  //stop the BFM
  cclo.stop();

  MPI_Barrier(MPI_COMM_WORLD);

  if (own_role == DLRM_AGG_ROLE)
  {
    for (unsigned int i = 0; i < count; ++i) {
      int res = (host_res_buf)[i];
      int ref = (int)i*numEmbedNodes;
      if (res != ref) {
        // std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
        //                 std::to_string(res) + " != " + std::to_string(ref) + ")"
        //           << std::endl;
        errors += 1;
      }
    }
    if (errors > 0) {
      std::cout << std::to_string(errors) + " errors!" << std::endl;
      failed_tests++;
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }
}
  

void start_test_sim(options_t options) {
  std::vector<rank_t> ranks = {};
  failed_tests = 0;
  skipped_tests = 0;

	std::vector<std::string> ipList;
  for (int i = 0; i < size; ++i) {
      rank_t new_rank = {"127.0.0.1", options.start_port + i, i, options.rxbuf_size};
      ranks.emplace_back(new_rank);
  }
	
  std::unique_ptr<ACCL::ACCL> accl;

  xrt::device device;

  accl = std::make_unique<ACCL::ACCL>(ranks, rank, options.start_port,
                                                options.udp ? networkProtocol::UDP : networkProtocol::TCP, options.num_rxbufmem,
                                                options.rxbuf_size);


  if (options.tcp){
    debug("Starting connections to communicator ranks");
    debug("Opening ports to communicator ranks");
    accl->open_port();
    MPI_Barrier(MPI_COMM_WORLD);
    debug("Starting session to communicator ranks");
    accl->open_con();
    test_debug(accl->dump_communicator(), options);
  }

  accl->set_timeout(1e6);

	MPI_Barrier(MPI_COMM_WORLD);
	unsigned int numEmbedNodes = (size-2)/2;
	test_dlrm_sim(*accl, options, numEmbedNodes);

  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << failed_tests << " tests failed on rank " << rank;
  if (skipped_tests > 0) {
    std::cout << " (skipped " << skipped_tests << " tests)";
  }
  std::cout << "." << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  if (failed_tests > 1) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}


// Total node in the system: 2*numEmbedNodes+2
// Reduce node rank(0) to rank(numEmbedNodes) with root at rank(numEmbedNodes)
// Embedding node rank(numEmbedNodes+1) to rank(2*numEmbedNodes)
// Final aggregate node rank(2*numEmbedNodes+1)
// Global communicator contains rank(0) to rank(2*numEmbedNodes+1)
// Reduction communicator contains rank(0) to rank(numEmbedNodes)

void start_test(options_t options) {
  std::vector<rank_t> ranks = {};
  failed_tests = 0;
  skipped_tests = 0;

  MPI_Barrier(MPI_COMM_WORLD);

	std::vector<std::string> ipList;

  std::ifstream myfile;
  myfile.open(options.fpgaIP);
  if(!myfile.is_open()) {
      perror("Error open fpgaIP file");
      exit(EXIT_FAILURE);
  }
  for (int i = 0; i < size; ++i) {
    std::string ip;
    getline(myfile, ip);
    ipList.push_back(ip);
    rank_t new_rank = {ip, options.start_port + i, i, options.rxbuf_size};
    ranks.emplace_back(new_rank);
  }

  unsigned int count = options.count;
  int errors = 0;
  if (size < 6) {
    std::cout<<"Error: Minimum group size is 6, current size:"<<size<<std::endl;
    return;
  }

  // figure out the local role
  unsigned int numEmbedNodes = (size-2)/2;
  unsigned int reduce_comm_size = numEmbedNodes+1;
  unsigned int reduce_comm_root = numEmbedNodes;
  std::cout << "Start dlrm_kernels hardware test with " <<numEmbedNodes<<" embedding nodes..."<< std::endl;

  unsigned int own_rank = rank;
	unsigned int own_role;
  std::string role_str;
	if ((own_rank > numEmbedNodes) && (own_rank < 2*numEmbedNodes+1)){
		own_role = DLRM_EMBED_ROLE;
    role_str="dlrm_embedding";
		std::cout<<"Rank:"<<own_rank<<" is DLRM_EMBED_ROLE"<<std::endl;
	} else if (own_rank<reduce_comm_size){
		if (own_rank == reduce_comm_root){
			own_role = DLRM_REDUCE_ROOT_ROLE;
      role_str="dlrm_reduce_root";
			std::cout<<"Rank:"<<own_rank<<" is DLRM_REDUCE_ROOT_ROLE"<<std::endl;
		} else {
			own_role = DLRM_REDUCE_SLAVE_ROLE;
      role_str="dlrm_reduce_slave";
			std::cout<<"Rank:"<<own_rank<<" is DLRM_REDUCE_SLAVE_ROLE"<<std::endl;
		}
	} else if (own_rank == 2*numEmbedNodes+1){
		own_role = DLRM_AGG_ROLE;
    role_str="dlrm_agg";
		std::cout<<"Rank:"<<own_rank<<" is DLRM_AGG_ROLE"<<std::endl;
	} else{
		std::cout<<"Error: rank not assigned to a role!"<<std::endl;
    return;
	}

  std::unique_ptr<ACCL::ACCL> accl;

  xrt::device device;

  device = xrt::device(options.device_index);
  
  std::string cclo_id;
  cclo_id = "0";
  
  std::string xclbin_str = options.xclbin+"/link_tcp_"+role_str+"_eth_0_debug_none_xilinx_u55c_gen3x16_xdma_3_202210_1/ccl_offload.xclbin";
  std::cout<<"XCLBIN:"<<xclbin_str<<std::endl;
  auto xclbin_uuid = device.load_xclbin(xclbin_str);
  auto cclo_ip = xrt::ip(device, xclbin_uuid,
                          "ccl_offload:{ccl_offload_" + cclo_id + "}");
  auto hostctrl_ip =
      xrt::kernel(device, xclbin_uuid, "hostctrl:{hostctrl_" + cclo_id + "_0}",
                  xrt::kernel::cu_access_mode::exclusive);

  int devicemem;
  std::vector<int> rxbufmem;
  int networkmem;
  devicemem = 0;
  for (int i=0; i<(int)options.num_rxbufmem; i++)
  {
    if(i<5){
      rxbufmem.push_back(i+1);
    }
  }
  networkmem = 6;
  
  if (options.tcp)
  {
    std::cout << "Configure TCP Network Kernel" << std::endl;
    auto network_krnl = xrt::kernel(device, xclbin_uuid, "network_krnl:{network_krnl_0}",
                  xrt::kernel::cu_access_mode::exclusive);
    
    uint localFPGAIP = ip_encode(ipList[rank]);
    std::cout << "rank: "<< rank << " FPGA IP: "<<std::hex << localFPGAIP << std::endl;

    auto tx_buf_network = xrt::bo (device, 8*1024*1024*sizeof(int8_t), networkmem);
    tx_buf_network.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto rx_buf_network = xrt::bo (device, 8*1024*1024*sizeof(int8_t), networkmem);
    rx_buf_network.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    network_krnl(localFPGAIP, uint(rank), localFPGAIP, tx_buf_network, rx_buf_network);

    uint32_t ip_reg = network_krnl.read_register(0x010);
    uint32_t board_reg = network_krnl.read_register(0x018);
    uint64_t ptr0 = network_krnl.read_register(0x028);
    uint64_t ptr1 = network_krnl.read_register(0x034);
    std::cout<< std::hex << "ip_reg: "<< ip_reg << " board_reg IP: " << board_reg <<" tx_ptr:"<<ptr0<<" rx_ptr:"<<ptr1<<std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  accl = std::make_unique<ACCL::ACCL>(
      ranks, rank, device, cclo_ip, hostctrl_ip, devicemem, rxbufmem,
      options.udp ? networkProtocol::UDP : networkProtocol::TCP,
      options.num_rxbufmem, options.rxbuf_size);
  
  accl->set_timeout(1e12);

  MPI_Barrier(MPI_COMM_WORLD);

  if (options.tcp){
    debug("Starting connections to communicator ranks");
    debug("Opening ports to communicator ranks");
    accl->open_port();
    MPI_Barrier(MPI_COMM_WORLD);
    debug("Starting session to communicator ranks");
    accl->open_con();
    debug(accl->dump_communicator());
  }

  accl->set_timeout(1e6);

  // MPI_Barrier(MPI_COMM_WORLD);
  // test_allreduce(*accl, options, reduceFunction::SUM);
  // MPI_Barrier(MPI_COMM_WORLD);
  // test_reduce_put(*accl, options, reduce_comm_root, reduceFunction::SUM);
  // MPI_Barrier(MPI_COMM_WORLD);

  std::string kernel_str=role_str+":{"+role_str+"_0_0}";
  std::cout<<"User kernel:"<<kernel_str<<std::endl;
  auto user_kernel = xrt::kernel(device, device.get_xclbin_uuid(), kernel_str,
                    xrt::kernel::cu_access_mode::exclusive);

	MPI_Barrier(MPI_COMM_WORLD);
	  
  // create reduce group from rank 0 to rank numEmbedNodes
	auto group = accl->get_comm_group(GLOBAL_COMM);
  std::vector<rank_t> reduce_group;
  communicatorId REDUCE_COMM;
  if (own_role == DLRM_REDUCE_ROOT_ROLE || own_role == DLRM_REDUCE_SLAVE_ROLE)
  {
    for (unsigned int i = 0; i < reduce_comm_size; i++)
    {
      reduce_group.push_back(group[i]);
    }
    REDUCE_COMM = accl->create_communicator(reduce_group, own_rank);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  debug(accl->dump_communicator());

  MPI_Barrier(MPI_COMM_WORLD);
	// create buffers
  int host_res_buf[options.count];

  for (unsigned int i = 0; i < count; i++) {
    host_res_buf[i] = 0;
  }

  xrt::bo res_buf_bo;

  xrt::bo   HBM_embedding0_bo,
            HBM_embedding1_bo,
            HBM_embedding2_bo,
            HBM_embedding3_bo,
            HBM_embedding4_bo,
            HBM_embedding5_bo,
            HBM_embedding6_bo,
            HBM_embedding7_bo;
  
  if (own_role == DLRM_EMBED_ROLE){

    std::cout<<"HBM_BANK0_SIZE:"<<HBM_BANK0_SIZE<<std::endl;

    size_t HBM_embedding0_size =  HBM_BANK0_SIZE;
    size_t HBM_embedding1_size =  HBM_BANK1_SIZE;
    size_t HBM_embedding2_size =  HBM_BANK2_SIZE;
    size_t HBM_embedding3_size =  HBM_BANK3_SIZE;
    size_t HBM_embedding4_size =  HBM_BANK4_SIZE;
    size_t HBM_embedding5_size =  HBM_BANK5_SIZE;
    size_t HBM_embedding6_size =  HBM_BANK6_SIZE;
    size_t HBM_embedding7_size =  HBM_BANK7_SIZE;

    int HBM_embedding0[HBM_BANK0_SIZE];
    int HBM_embedding1[HBM_BANK1_SIZE];
    int HBM_embedding2[HBM_BANK2_SIZE];
    int HBM_embedding3[HBM_BANK3_SIZE];
    int HBM_embedding4[HBM_BANK4_SIZE];
    int HBM_embedding5[HBM_BANK5_SIZE];
    int HBM_embedding6[HBM_BANK6_SIZE];
    int HBM_embedding7[HBM_BANK7_SIZE];

    const int weights_init = 1;
    for (int i = 0 ; i < HBM_embedding0_size ; i++) {
      HBM_embedding0[i] = weights_init;
    }
    for (int i = 0 ; i < TABLE_SIZE_HBM_1 ; i++) {
      HBM_embedding1[i] = weights_init;
    }
    for (int i = 0 ; i < TABLE_SIZE_HBM_2 ; i++) {
      HBM_embedding2[i] = weights_init;
    }
    for (int i = 0 ; i < TABLE_SIZE_HBM_3 ; i++) {
      HBM_embedding3[i] = weights_init;
    }
    for (int i = 0 ; i < TABLE_SIZE_HBM_4 ; i++) {
      HBM_embedding4[i] = weights_init;
    }
    for (int i = 0 ; i < TABLE_SIZE_HBM_5 ; i++) {
      HBM_embedding5[i] = weights_init;
    }
    for (int i = 0 ; i < TABLE_SIZE_HBM_6 ; i++) {
      HBM_embedding6[i] = weights_init;
    }
    for (int i = 0 ; i < TABLE_SIZE_HBM_7 ; i++) {
      HBM_embedding7[i] = weights_init;
    }

    std::cout<<"HBM embedding init done"<<std::endl;

    HBM_embedding0_bo = xrt::bo(device, HBM_embedding0_size *sizeof(int), user_kernel.group_id(0));
    HBM_embedding0_bo.write(HBM_embedding0);
    HBM_embedding0_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    HBM_embedding1_bo = xrt::bo(device, HBM_embedding1_size *sizeof(int), user_kernel.group_id(1));
    HBM_embedding1_bo.write(HBM_embedding1);
    HBM_embedding1_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    HBM_embedding2_bo = xrt::bo(device, HBM_embedding2_size *sizeof(int), user_kernel.group_id(2));
    HBM_embedding2_bo.write(HBM_embedding2);
    HBM_embedding2_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    HBM_embedding3_bo = xrt::bo(device, HBM_embedding3_size *sizeof(int), user_kernel.group_id(3));
    HBM_embedding3_bo.write(HBM_embedding3);
    HBM_embedding3_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    HBM_embedding4_bo = xrt::bo(device, HBM_embedding4_size *sizeof(int), user_kernel.group_id(4));
    HBM_embedding4_bo.write(HBM_embedding4);
    HBM_embedding4_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    HBM_embedding5_bo = xrt::bo(device, HBM_embedding5_size *sizeof(int), user_kernel.group_id(5));
    HBM_embedding5_bo.write(HBM_embedding5);
    HBM_embedding5_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    HBM_embedding6_bo = xrt::bo(device, HBM_embedding6_size *sizeof(int), user_kernel.group_id(6));
    HBM_embedding6_bo.write(HBM_embedding6);
    HBM_embedding6_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    HBM_embedding7_bo = xrt::bo(device, HBM_embedding7_size *sizeof(int), user_kernel.group_id(7));
    HBM_embedding7_bo.write(HBM_embedding7);
    HBM_embedding7_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "HBM_embedding allocation Finish" << std::endl; 

  } else if (own_role == DLRM_REDUCE_ROOT_ROLE)
  {
    // do nothing
  } else if (own_role == DLRM_REDUCE_SLAVE_ROLE)
  {
    // do nothing
  } else if (own_role == DLRM_AGG_ROLE)
  {
    res_buf_bo = xrt::bo(device, sizeof(int)*options.count, user_kernel.group_id(0));
    res_buf_bo.write(host_res_buf);
    res_buf_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "Start user kernels" << std::endl;

  // Embedding nodes
  if (own_role == DLRM_EMBED_ROLE)
  {
    unsigned int dst_rank = own_rank - reduce_comm_size;
    auto run = user_kernel(
        HBM_embedding0_bo,
        HBM_embedding1_bo,
        HBM_embedding2_bo,
        HBM_embedding3_bo,
        HBM_embedding4_bo,
        HBM_embedding5_bo,
        HBM_embedding6_bo,
        HBM_embedding7_bo,
        dst_rank, own_rank, size, accl->get_communicator_addr(GLOBAL_COMM), accl->get_arithmetic_config_addr({dataType::int32, dataType::int32}));
    run.wait();
  }
  // recv and reduce operation for reduce nodes
  else if (own_role == DLRM_REDUCE_ROOT_ROLE)
  {
    unsigned int destination = 2*numEmbedNodes+1;
    auto run = user_kernel(destination, reduce_comm_root, (unsigned int)ACCL::reduceFunction::SUM, own_rank, size, accl->get_communicator_addr(GLOBAL_COMM), accl->get_communicator_addr(REDUCE_COMM), accl->get_arithmetic_config_addr({dataType::int32, dataType::int32}));
    run.wait();
  } else if (own_role == DLRM_REDUCE_SLAVE_ROLE)
  {
    auto run = user_kernel(reduce_comm_root, (unsigned int)ACCL::reduceFunction::SUM, own_rank, size, accl->get_communicator_addr(GLOBAL_COMM), accl->get_communicator_addr(REDUCE_COMM), accl->get_arithmetic_config_addr({dataType::int32, dataType::int32}));
    run.wait();
  } else if (own_role == DLRM_AGG_ROLE)
  {
    auto run = user_kernel(res_buf_bo, own_rank, size, accl->get_communicator_addr(GLOBAL_COMM), accl->get_arithmetic_config_addr({dataType::int32, dataType::int32}));
    run.wait();
  }

  std::cout << "Finish user kernels" << std::endl;
  
  if (own_role == DLRM_AGG_ROLE)
  {
    res_buf_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    res_buf_bo.read(host_res_buf);

    for (unsigned int i = 0; i < count; ++i) {
      int res = (host_res_buf)[i];
      int ref = (int)i*numEmbedNodes;
      if (res != ref) {
        // std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
        //                 std::to_string(res) + " != " + std::to_string(ref) + ")"
        //           << std::endl;
        errors += 1;
      }
    }
    if (errors > 0) {
      std::cout << std::to_string(errors) + " errors!" << std::endl;
      failed_tests++;
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }

  std::cout << failed_tests << " tests failed on rank " << rank;
  if (skipped_tests > 0) {
    std::cout << " (skipped " << skipped_tests << " tests)";
  }
  std::cout << "." << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  // Enable the benchmarking
  if (own_role == DLRM_AGG_ROLE)
  {
    std::cout << "Enable hw bench kernel" << std::endl;
    auto hw_bench_krnl = xrt::kernel(device, device.get_xclbin_uuid(), "collector:{collector_0}",xrt::kernel::cu_access_mode::exclusive);  

    std::cout << "Allocate Buffer in Global Memory\n";
    auto buf_hw_bench_cmd = xrt::bo (device, 8*1024*1024*sizeof(uint64_t), hw_bench_krnl.group_id(1));
    auto buf_hw_bench_sts = xrt::bo (device, 8*1024*1024*sizeof(uint64_t), hw_bench_krnl.group_id(2));

    auto host_ptr_hw_bench_cmd = buf_hw_bench_cmd.map<uint64_t*>();
    auto host_ptr_hw_bench_sts = buf_hw_bench_sts.map<uint64_t*>();

    std::fill(host_ptr_hw_bench_cmd, host_ptr_hw_bench_cmd + 8*1024*1024, 0);
    std::fill(host_ptr_hw_bench_sts, host_ptr_hw_bench_sts + 8*1024*1024, 0);

    std::cout << "synchronize input buffer data to device global memory\n";
    buf_hw_bench_cmd.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buf_hw_bench_sts.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    int hw_bench_record = CONFIG_HW_BENCH_RECORD + (size-1) + 1;

    std::cout << "Execution of the kernel\n";
    auto run = hw_bench_krnl((hw_bench_record), buf_hw_bench_cmd, buf_hw_bench_sts);
    uint32_t round_reg = hw_bench_krnl.read_register(0x010);
    uint32_t cmd_addr_reg = hw_bench_krnl.read_register(0x018);
    uint32_t sts_addr_reg = hw_bench_krnl.read_register(0x024);
    std::cout<< std::hex << "round_reg: "<< round_reg <<" cmd_addr_reg: "<<cmd_addr_reg<<" sts_addr_reg: "<<sts_addr_reg<<std::endl;
      
    run.wait(1000);
    
    std::cout <<"Sync hw_bench mem to host"<< std::endl;
    
    buf_hw_bench_cmd.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    unsigned int cmd_mem_offset = 0;
    buf_hw_bench_sts.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    unsigned int sts_mem_offset = 0;
    for (unsigned int i = 0; i < hw_bench_record; i++)
    {
      timestamp_t timestamp = readTimeStamp(host_ptr_hw_bench_cmd, host_ptr_hw_bench_sts, cmd_mem_offset, sts_mem_offset);
      printTimeStamp(timestamp, options);
    }
  }

  // debug(accl->dump_communicator());

  if (failed_tests > 1) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}


bool xrt_simulator_ready(const options_t &opts) {
  if (opts.hardware) {
    return true;
  }

  const char *vitis = std::getenv("XILINX_VITIS");

  if (vitis == nullptr) {
    return false;
  }

  const char *emu = std::getenv("XCL_EMULATION_MODE");
  if (emu == nullptr) {
    return false;
  }

  return std::string(emu) == "sw_emu" || std::string(emu) == "hw_emu";
}

options_t parse_options(int argc, char *argv[]) {
    try {
    TCLAP::CmdLine cmd("Test ACCL C++ driver");
    TCLAP::ValueArg<unsigned int> nruns_arg("n", "nruns",
                                            "How many times to run each test",
                                            false, 1, "positive integer");
    cmd.add(nruns_arg);
    TCLAP::ValueArg<uint16_t> start_port_arg(
        "s", "start-port", "Start of range of ports usable for sim", false, 5500,
        "positive integer");
    cmd.add(start_port_arg);
    TCLAP::ValueArg<uint32_t> count_arg("c", "count", "How many element per buffer",
                                        false, (BATCH_NUM * BATCH_SIZE * 3 * 64 * 16), "positive integer");
    cmd.add(count_arg);
    TCLAP::ValueArg<uint16_t> bufsize_arg("b", "rxbuf-size",
                                          "How many KB per RX buffer", false, 256,
                                          "positive integer");
    cmd.add(bufsize_arg);
    TCLAP::ValueArg<uint32_t> seg_arg("g", "max_segment_size",
                                          "Maximum segmentation size in KB (should be samller than Max DMA transaction)", false, 4096,
                                          "positive integer");
    cmd.add(seg_arg);
    TCLAP::ValueArg<uint16_t> num_rxbufmem_arg ("m", "num_rxbufmem",
                                          "Number of memory banks used for rxbuf", false, 4,
                                          "positive integer");
    cmd.add(num_rxbufmem_arg);
    TCLAP::ValueArg<uint16_t> test_mode_arg ("y", "test_mode",
                                          "Test mode, by default run all the collective tests", false, 0,
                                          "integer");
    cmd.add(test_mode_arg);
    TCLAP::SwitchArg debug_arg("d", "debug", "Enable debug mode", cmd, false);
    TCLAP::SwitchArg hardware_arg("f", "hardware", "enable hardware mode", cmd,
                                  false);
    TCLAP::SwitchArg axis3_arg("a", "axis3", "Use axis3 hardware setup", cmd,
                              false);
    TCLAP::SwitchArg udp_arg("u", "udp", "Use UDP hardware setup", cmd, false);
    TCLAP::SwitchArg tcp_arg("t", "tcp", "Use TCP hardware setup", cmd, false);
    TCLAP::SwitchArg hwbench_arg("z", "hwbench", "Enable hwbench, the maximum CCLO commands (~20) is limited by the FIFO depth to the bench kernel", cmd, false);
    TCLAP::SwitchArg userkernel_arg("k", "userkernel", "Enable user kernel(by default vadd kernel)", cmd, false);
    TCLAP::ValueArg<std::string> xclbin_arg(
        "x", "xclbin", "xclbin of accl driver if hardware mode is used", false,
        "accl.xclbin", "file");
    cmd.add(xclbin_arg);
    TCLAP::ValueArg<std::string> fpgaIP_arg(
        "l", "ipList", "ip list of FPGAs if hardware mode is used", false,
        "fpga", "file");
    cmd.add(fpgaIP_arg);
    TCLAP::ValueArg<uint16_t> device_index_arg(
        "i", "device-index", "device index of FPGA if hardware mode is used",
        false, 0, "positive integer");
    cmd.add(device_index_arg);
    cmd.parse(argc, argv);
    if (hardware_arg.getValue()) {
			if (axis3_arg.getValue()){
				if (udp_arg.getValue() || tcp_arg.getValue()){
					throw std::runtime_error("When using hardware axis3 mode, tcp or udp can not be used.");
				}
				std::cout << "Hardware axis3 mode" << std::endl;
			}  
			if (udp_arg.getValue())
			{
				if (axis3_arg.getValue() || tcp_arg.getValue()){
					throw std::runtime_error("When using hardware udp mode, tcp or axis3 can not be used.");
				}
				std::cout << "Hardware udp mode" << std::endl;
			}
			if (tcp_arg.getValue())
			{
				if (axis3_arg.getValue() || udp_arg.getValue()){
					throw std::runtime_error("When using hardware tcp mode, udp or axis3 can not be used.");
				}
				std::cout << "Hardware tcp mode" << std::endl;
			}
      if ((axis3_arg.getValue() || udp_arg.getValue() || tcp_arg.getValue()) == false) {
        throw std::runtime_error("When using hardware, specify either axis3 or tcp or"
                                 " udp mode.");
      }
      if (hwbench_arg.getValue() && hardware_arg.getValue()==false)
      {
        throw std::runtime_error("Hardware bench mode should be set with hardware mode.");
      }
      if(hwbench_arg.getValue() && (test_mode_arg.getValue()==0)){
        throw std::runtime_error("Hardware bench mode can not run will test mode ALL, run single collective bench.");
      }
    }

    options_t opts;
    opts.start_port = start_port_arg.getValue();
    opts.count = count_arg.getValue();
    opts.rxbuf_size = bufsize_arg.getValue() * 1024; // convert to bytes
    opts.seg_size = seg_arg.getValue() * 1024; // convert to bytes
    opts.num_rxbufmem = num_rxbufmem_arg.getValue();
    opts.nruns = nruns_arg.getValue();
    opts.debug = debug_arg.getValue();
    opts.hardware = hardware_arg.getValue();
    opts.axis3 = axis3_arg.getValue();
    opts.udp = udp_arg.getValue();
    opts.tcp = tcp_arg.getValue();
    opts.test_mode = test_mode_arg.getValue();
    opts.hw_bench = hwbench_arg.getValue();
    opts.enableUserKernel = userkernel_arg.getValue();
    opts.device_index = device_index_arg.getValue();
    opts.xclbin = xclbin_arg.getValue();
    opts.fpgaIP = fpgaIP_arg.getValue();
    opts.test_xrt_simulator = xrt_simulator_ready(opts);

    std::cout<<"count:"<<opts.count<<" rxbuf_size:"<<opts.rxbuf_size<<" seg_size:"<<opts.seg_size<<" num_rxbufmem:"<<opts.num_rxbufmem<<std::endl;
    return opts;

  } catch (std::exception &e) {
    if (rank == 0) {
      std::cout << "Error: " << e.what() << std::endl;
    }

    MPI_Finalize();
    exit(1);
  }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    options_t options = parse_options(argc, argv);

    int len;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(name, &len);

    std::ostringstream stream;
    stream << prepend_process() << "rank " << rank << " size " << size <<" "<< name
          << std::endl;
    std::cout << stream.str();

    MPI_Barrier(MPI_COMM_WORLD);

    if(options.hardware){
      std::cout<<"Hardware Test"<<std::endl;
      start_test(options);
    } else {
      start_test_sim(options);
    }
    
    MPI_Finalize();
    return 0;
}
