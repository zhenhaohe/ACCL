#include "dlrm.h"

int main()
{

    unsigned int destination = 1;
    unsigned int root = 0;
    unsigned int function = 0;
    ap_uint<32> local_rank = 0;
    ap_uint<32> global_comm_size = 2;
    ap_uint<32> global_comm_adr = 0;
    ap_uint<32> reduce_comm_adr = 0;
    ap_uint<32> dpcfg_adr = 0;

    int dlrm_one_inference_count_send = (2 * 64 + 16) * 16;
    int dlrm_one_inference_count_recv = 2 * 64 * 16;
    int num_inference = BATCH_NUM * BATCH_SIZE;
    int dlrm_count_send = num_inference * dlrm_one_inference_count_send;
    int dlrm_count_recv = num_inference * dlrm_one_inference_count_recv;

    hls::stream<command_word> cmd_to_cclo, sts_from_cclo;
	hls::stream<stream_word> data_to_cclo, data_from_cclo;

    // data received from barrier
    stream_word tmp_word;
    tmp_word.data = 0;
    data_from_cclo.write(tmp_word);

    command_word sts_word;
    sts_word.data = 1;

    sts_from_cclo.write(sts_word); // pop in barrier status

    for (int j = 0; j < num_inference; j++)
    {
      sts_from_cclo.write(sts_word); // pop in send status
    }

    for (int i = 0; i< dlrm_count_recv/16; i++)
    {
        data_from_cclo.write(tmp_word); // data received from reduce slave node
    }
    
    dlrm_reduce_root(
        destination,
        //reduce configuration
        root,
        function,
        //parameters pertaining to CCLO config
        local_rank,
        global_comm_size,
        global_comm_adr, 
        reduce_comm_adr, 
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

    for (int j = 0; j < num_inference; j++)
    {
        for (int i = 0; i< 15; i++)
        {
            command_word cmd = cmd_to_cclo.read(); // read dlrm send
            std::cout<<"dlrm send cmd:"<<cmd.data<<std::endl;
        }
    }

    for (int i =0; i< dlrm_count_send/16; i++)
    {   
        data_to_cclo.read();
        std::cout<<"dlrm send data "<<i<<std::endl;
    }

    return 0;
}
