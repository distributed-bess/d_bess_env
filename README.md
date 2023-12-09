# d_bess_env
The distributed ECs (BESSs) environment
* The ECs' power data are deleted because it's corporate property. You can use the public data set of servers to test the environment. The power data is named by EC-0.txt, EC-1.txt, etc. The length of it is 672, indicating one week time.
* The dod function data is in the data/dod_data comprising LA and LFP battery.
* The functions of single BESS are implemented in [onebess.py](\env\one_bess.py)
* The functions of distributed BESS are implemented in [bess_d.py](\env\bess_d.py)
* The config file of distributed BESS env is <config/bess_d.yaml>, which consists of the setting parameters of the env.
* You can find the implementation of DQN from the RLlib or other existing projects.
* The implementation of PER can be found at [PER](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py)
  
