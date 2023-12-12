# d_bess_env
The distributed ECs (BESSs) environment and instructions:
* The power data of the ECs has been deleted as it is considered proprietary information. To test the environment, you can utilize the public dataset of servers. The power data files are named as EC-0.txt, EC-1.txt, etc. in folder `data\power_data`, with a length of 672 indicating a one-week timeframe.
* The data for the dod function is contained within the folder `data\dod_data`, which includes both LA and LFP batteries. Moreover, you must install `xlrd 1.2.0` to read these files.
* The functionalities of single BESS are implemented in [onebess.py](d_bess_env/env/onebess.py).
* The functionalities of distributed BESS are implemented in [bess_d.py](d_bess_env/env/bess_d.py).
* The configuration file for the distributed BESS environment is located at [bess_d.yaml](d_bess_env/env/config/bess_d.yaml), which contains the parameter settings for the environment and the batteries. Specifically, "powers_data": "0-3" indicates that the power data corresponds to EC-0.txt, EC-1.txt, and EC-2.txt.
* The implementation of DQN can be found in RLlib or other established projects.
* You can use the invalid action mask as `q_values[avail_actions == 0.0] = -float("inf")` and the mask can be obtained by function `get_avail_agent_actions()`.
* The implementation of PER can be found at [PER](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py).
  
