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

#include "accl/coyotedevice.hpp"
#include "accl/common.hpp"
#include "cProcess.hpp"
#include <future>
#include <iomanip>

// static void finish_coyote_request(ACCL::CoyoteRequest *req) {
//   req->wait_kernel();
//   ACCL::CoyoteDevice *cclo = reinterpret_cast<ACCL::CoyoteDevice *>(req->cclo());
//   // get ret code before notifying waiting threads
//   req->set_retcode(cclo->read(ACCL::CCLO_ADDR::RETCODE_OFFSET));
//   req->set_duration(cclo->read(ACCL::CCLO_ADDR::PERFCNT_OFFSET));
//   req->set_status(ACCL::operationStatus::COMPLETED);
//   req->notify();
//   cclo->complete_request(req);
// }

namespace ACCL {

// void CoyoteRequest::start() {
//   std::cout<<"CoyoteRequest start not needed in Coyote ACCL"<<std::endl;
//   // assert(this->get_status() ==  operationStatus::EXECUTING);

//   // int function, arg_id = 0;

//   // if (options.scenario == operation::config) {
//   //   function = static_cast<int>(options.cfg_function);
//   // } else {
//   //   function = static_cast<int>(options.reduce_function);
//   // }
//   // uint32_t flags = static_cast<uint32_t>(options.host_flags) << 8 | static_cast<uint32_t>(options.stream_flags);

//   // auto coyote_proc = reinterpret_cast<ACCL::CoyoteDevice *>(cclo())->get_device();

//   // if (coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2) && (0x02 == 0)) { // read AP_CTRL and check bit 2 (the done bit)
//   //   throw std::runtime_error(
//   //       "Error, collective is already running, wait for previous to complete!");
//   // }

//   // coyote_proc->setCSR(static_cast<uint32_t>(options.scenario), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::SCEN)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(options.count), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::LEN)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(options.comm), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::COMM)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(options.root_src_dst), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ROOT_SRC_DST)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(function), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::FUNCTION_R)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(options.tag), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::MSG_TAG)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(options.arithcfg_addr), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::DATAPATH_CFG)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(options.compression_flags), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::COMPRESSION_FLAGS)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(flags), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::STREAM_FLAGS)>>2);
//   // addr_t addr_a = options.addr_0->address();
//   // coyote_proc->setCSR(static_cast<uint32_t>(addr_a), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRA_0)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(addr_a >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRA_1)>>2);
//   // addr_t addr_b = options.addr_1->address();
//   // coyote_proc->setCSR(static_cast<uint32_t>(addr_b), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRB_0)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(addr_b >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRB_1)>>2);
//   // addr_t addr_c = options.addr_2->address();
//   // coyote_proc->setCSR(static_cast<uint32_t>(addr_c), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRC_0)>>2);
//   // coyote_proc->setCSR(static_cast<uint32_t>(addr_c >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRC_1)>>2);

//   // auto f = std::async(std::launch::async, finish_coyote_request, this);

//   // // start the kernel
//   // coyote_proc->setCSR(0x1U, (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);

// }

// void CoyoteRequest::wait_kernel() {
//   std::cout<<"CoyoteRequest wait_kernel not needed in Coyote ACCL"<<std::endl;
//   // auto coyote_proc = reinterpret_cast<ACCL::CoyoteDevice *>(cclo())->get_device();
//   // uint32_t is_done = 0;
//   // double durationUs = 0.0;
// 	// auto start = std::chrono::high_resolution_clock::now();
//   // while (!is_done) {
//   //   uint32_t regi = coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);
//   //   is_done = (regi >> 1) & 0x1; // get bit 1 of AP_CTRL register
//   //   auto end = std::chrono::high_resolution_clock::now();
// 	// 	durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
//   //   if (durationUs > 500000.0)
//   //     break;
//   // }
// }

CoyoteDevice::CoyoteDevice(): num_qp(0) {
  this->coyote_proc = new fpga::cProcess(targetRegion, getpid());
	std::cerr << "ACLL DEBUG: aquiring cProc: targetRegion: " << targetRegion << ", cPid: " << coyote_proc->getCpid() << std::endl;
}

CoyoteDevice::CoyoteDevice(unsigned int num_qp): num_qp(num_qp) {

  for (unsigned int i=0; i<(num_qp+1); i++)
  {
    fpga::cProcess* cproc = new fpga::cProcess(targetRegion, getpid());
    coyote_qProc_vec.push_back(cproc);
  }

  for (unsigned int i=0; i<coyote_qProc_vec.size(); i++){
    if(coyote_qProc_vec[i]->getCpid() == 0){
      this->coyote_proc = coyote_qProc_vec[i];
      std::cerr << "ACLL DEBUG: aquiring cProc: targetRegion: " << targetRegion << ", cPid: " << coyote_proc->getCpid() << std::endl;
      coyote_qProc_vec.erase(coyote_qProc_vec.begin() + i);
      break;
    }
  }

  if(coyote_proc == NULL || coyote_proc->getCpid() != 0){
    std::cerr << "cProc initialization error!"<<std::endl;

  }

  for (unsigned int i=0; i<coyote_qProc_vec.size(); i++){
    std::cerr << "ACLL DEBUG: aquiring qProc: targetRegion: " << targetRegion << ", cPid: " << coyote_qProc_vec[i]->getCpid() << std::endl;
  }

}

ACCLRequest *CoyoteDevice::start(const Options &options) {
  int function, arg_id = 0;

  if (options.scenario == operation::config) {
    function = static_cast<int>(options.cfg_function);
  } else {
    function = static_cast<int>(options.reduce_function);
  }
  uint32_t flags = static_cast<uint32_t>(options.host_flags) << 8 | static_cast<uint32_t>(options.stream_flags);

  if (coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2) && (0x02 == 0)) { // read AP_CTRL and check bit 2 (the done bit)
    throw std::runtime_error(
        "Error, collective is already running, wait for previous to complete!");
  }

  coyote_proc->setCSR(static_cast<uint32_t>(options.scenario), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::SCEN)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.count), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::LEN)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.comm), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::COMM)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.root_src_dst), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ROOT_SRC_DST)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(function), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::FUNCTION_R)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.tag), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::MSG_TAG)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.arithcfg_addr), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::DATAPATH_CFG)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(options.compression_flags), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::COMPRESSION_FLAGS)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(flags), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::STREAM_FLAGS)>>2);
  addr_t addr_a = options.addr_0->address();
  coyote_proc->setCSR(static_cast<uint32_t>(addr_a), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRA_0)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(addr_a >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRA_1)>>2);
  addr_t addr_b = options.addr_1->address();
  coyote_proc->setCSR(static_cast<uint32_t>(addr_b), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRB_0)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(addr_b >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRB_1)>>2);
  addr_t addr_c = options.addr_2->address();
  coyote_proc->setCSR(static_cast<uint32_t>(addr_c), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRC_0)>>2);
  coyote_proc->setCSR(static_cast<uint32_t>(addr_c >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRC_1)>>2);

  // start the kernel
  coyote_proc->setCSR(0x1U, (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);

  return nullptr;

}

void CoyoteDevice::wait(ACCLRequest *request) { 
  uint32_t is_done = 0;
  uint32_t last = 0xffffffff;
  while (!is_done) {
    uint32_t regi = coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);
    if (last != regi) {
      // std::cerr << "Read from AP_CTRL: " << std::setbase(16) << regi << std::setbase(10) << std::endl;
      last = regi;
    }
    is_done = (regi >> 1) & 0x1;
  }
}

timeoutStatus CoyoteDevice::wait(ACCLRequest *request,
                               std::chrono::milliseconds timeout) {
  uint32_t is_done = 0;
  uint32_t last = 0xffffffff;
  auto start = std::chrono::high_resolution_clock::now();
  while (!is_done) {
    uint32_t regi = coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);
    if (last != regi) {
      // std::cerr << "Read from AP_CTRL: " << std::setbase(16) << regi << std::setbase(10) << std::endl;
      last = regi;
    }
    is_done = (regi >> 1) & 0x1;
    auto end = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(end-start) > timeout){
        std::cout<<"CoyoteDevice Wait Time Out"<<std::endl;
        break;
    } 
  }

  return timeoutStatus::no_timeout;
}

CCLO::deviceType CoyoteDevice::get_device_type()
{
  std::cerr<<"get_device_type: coyote_device"<<std::endl;
  return CCLO::coyote_device;
}

void CoyoteDevice::printDebug(){
  coyote_proc->printDebug();

  std::ifstream inputFile("/sys/kernel/coyote_cnfg/cyt_attr_nstats_q0");  

  if (!inputFile.is_open()) {
      std::cerr << "Failed to open net sts file." << std::endl;
  }

  // Read and print the file line by line
  std::string line;
  while (std::getline(inputFile, line)) {
      std::cout << line << std::endl;
  }

  // Close the file
  inputFile.close();
}

bool CoyoteDevice::test(ACCLRequest *request) {
  std::cout<<"CoyoteDevice test not needed in Coyote ACCL"<<std::endl;
  return false;
  // auto fpga_handle = request_map.find(*request);

  // if (fpga_handle == request_map.end())
  //   return true;

  // return fpga_handle->second->get_status() == operationStatus::COMPLETED;
}

uint64_t CoyoteDevice::get_duration(ACCLRequest *request) {  
  return (read(CCLO_ADDR::PERFCNT_OFFSET) * 4);
}

void CoyoteDevice::free_request(ACCLRequest *request) {
  std::cout<<"CoyoteDevice free_request not needed in Coyote ACCL"<<std::endl;
  // auto fpga_handle = request_map.find(*request);

  // if (fpga_handle != request_map.end()) {
  //   delete fpga_handle->second;
  //   request_map.erase(fpga_handle);
  // }
}

val_t CoyoteDevice::get_retcode(ACCLRequest *request) {
  return read(CCLO_ADDR::RETCODE_OFFSET);
}

ACCLRequest *CoyoteDevice::call(const Options &options) {
  ACCLRequest *req = start(options);
  wait(req);
  
  // internal use only
  return req;
}

val_t CoyoteDevice::read(addr_t offset) {
	// std::cerr << "CoyoteDevice read address: " << ((OFFSET_CCLO + offset)>>2) << std::endl;
  return coyote_proc->getCSR((OFFSET_CCLO + offset)>>2);
}

void CoyoteDevice::write(addr_t offset, val_t val) {
	// std::cerr << "CoyoteDevice write address: " << ((OFFSET_CCLO + offset)>>2) << std::endl;
  coyote_proc->setCSR(val, (OFFSET_CCLO + offset)>>2);
}

// void CoyoteDevice::launch_request() {
//   std::cout<<"CoyoteDevice launch_request not needed in Coyote ACCL"<<std::endl;
//   // // This guarantees permission to only one thread trying to start an operation
//   // if (queue.run()) {
//   //   CoyoteRequest *req = queue.front();
//   //   assert(req->get_status() == operationStatus::QUEUED);
//   //   req->set_status(operationStatus::EXECUTING);
//   //   req->start();
//   // }
// }

// void CoyoteDevice::complete_request(CoyoteRequest *request) {
//   std::cout<<"CoyoteDevice complete_request not needed in Coyote ACCL"<<std::endl;
//   // if (request->get_status() == operationStatus::COMPLETED) {
//   //   queue.pop();
//   //   launch_request();
//   // }
// }

} // namespace ACCL
