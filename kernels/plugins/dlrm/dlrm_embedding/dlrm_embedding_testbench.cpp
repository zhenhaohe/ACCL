#include "dlrm.h"

int main()
{

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
    for (int i = 0 ; i < HBM_embedding0_size ; i++) {
      HBM_embedding1[i] = weights_init;
    }
    for (int i = 0 ; i < HBM_embedding0_size ; i++) {
      HBM_embedding2[i] = weights_init;
    }
    for (int i = 0 ; i < HBM_embedding0_size ; i++) {
      HBM_embedding3[i] = weights_init;
    }
    for (int i = 0 ; i < HBM_embedding0_size ; i++) {
      HBM_embedding4[i] = weights_init;
    }
    for (int i = 0 ; i < HBM_embedding0_size ; i++) {
      HBM_embedding5[i] = weights_init;
    }
    for (int i = 0 ; i < HBM_embedding0_size ; i++) {
      HBM_embedding6[i] = weights_init;
    }
    for (int i = 0 ; i < HBM_embedding0_size ; i++) {
      HBM_embedding7[i] = weights_init;
    }

    std::cout<<"HBM embedding init done"<<std::endl;


    unsigned int destination = 1;
    ap_uint<32> local_rank = 0;
    ap_uint<32> comm_size = 2;
    ap_uint<32> comm_adr = 0;
    ap_uint<32> dpcfg_adr = 0;

    hls::stream<command_word> cmd_to_cclo, sts_from_cclo;
	hls::stream<stream_word> data_to_cclo, data_from_cclo;

    // data received from barrier
    stream_word tmp_word;
    tmp_word.data = 0;
    data_from_cclo.write(tmp_word);

    command_word sts_word;
    sts_word.data = 1;

    sts_from_cclo.write(sts_word); // pop in barrier status
    sts_from_cclo.write(sts_word); // pop in send status
    
    dlrm_embedding(
      // dlrm parameters
      HBM_embedding0,
      HBM_embedding1,
      HBM_embedding2,
      HBM_embedding3,
      HBM_embedding4,
      HBM_embedding5,
      HBM_embedding6,
      HBM_embedding7,
      destination,
      //parameters pertaining to CCLO config
      local_rank,
      comm_size,
      comm_adr, 
      dpcfg_adr,
      //streams to and from CCLO
      cmd_to_cclo,
      sts_from_cclo,
      data_to_cclo,
      data_from_cclo
    );

    for (int i = 0; i< 15; i++)
    {
        command_word cmd = cmd_to_cclo.read(); // read barrier send
        std::cout<<"dlrm barrer send cmd:"<<cmd.data<<std::endl;
    }
    
    data_to_cclo.read();
    std::cout<<"dlrm barrier data"<<std::endl;
 
    for (int i = 0; i< 15; i++)
    {
        command_word cmd = cmd_to_cclo.read(); // read dlrm send
        std::cout<<"dlrm send cmd:"<<cmd.data<<std::endl;
    }

    int dlrm_count = BATCH_NUM * BATCH_SIZE * 3 * 64 * 16;
    int dlrm_word_cnt = dlrm_count/16;
    for (int i =0; i< dlrm_word_cnt; i++)
    {   
        data_to_cclo.read();
        std::cout<<"dlrm send data "<<i<<std::endl;
    }

    return 0;
}
