#! /bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Script Dir: $SCRIPT_DIR"

echo "Compile"

cd $SCRIPT_DIR && make

num_process=0
rm $SCRIPT_DIR/host
rm $SCRIPT_DIR/fpga

# server IDs (u55c)
SERVID=(2 3 4 5 6 9)
for servid in ${SERVID[@]}; do 
    get_from_vars -n --filter alveo_u55c_$(printf "%02d" $servid)_mellanox_0 | awk -F"\"| " '{print $2}' >>$SCRIPT_DIR/host
    get_from_vars -n --filter alveo_u55c_$(printf "%02d" $servid)_fpga_0 | awk -F"\"| " '{print $2}' >>$SCRIPT_DIR/fpga
    num_process=$((num_process+1))
    hostlist+="alveo-u55c-$(printf "%02d" $servid) "
done

echo "START DLRM TEST"

NUM_ELE=(16384)
for np in `seq $num_process $num_process`; do
    for num_ele in ${NUM_ELE[@]}; do 
        mpirun -n $np --iface ens4f0 -prepend-rank -f ./host $SCRIPT_DIR/bin/test -d -f -t -b 4096 -c $num_ele -l $SCRIPT_DIR/fpga -x $SCRIPT_DIR/../../../test/hardware/ &
        sleeptime=$(((np-2) * 2+40))
        sleep $sleeptime
        parallel-ssh -H "$hostlist" "kill -9 \$(ps -aux | grep test | awk '{print \$2}')" 
        parallel-ssh -H "$hostlist" "/opt/xilinx/xrt/bin/xbutil reset --force --device 0000:c4:00.1"
    done
done

