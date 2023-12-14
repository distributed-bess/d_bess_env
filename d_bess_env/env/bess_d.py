# -*-coding:utf-8-*-
import numpy as np
import os, yaml
from .onebess import OneBESS, load_env_args
import math


def clip_float_2(n):
    return int(n * 100) / 100


class BESS_d():

    def __init__(self, kwargs_name="bess_3_d"):
        kwargs = load_env_args(kwargs_name)
        self.args = kwargs
        self.map_name = getattr(self.args, "map_name")
        self.bat_type = getattr(self.args, "bat_type")
        self.n_agents = getattr(self.args, "n_agents")
        self.delta_t = getattr(self.args, "delta_t")
        self.p_d = getattr(self.args, "demand_charge_price")
        self.p_e = getattr(self.args, "energy_charge_price")
        self.episode_days = getattr(self.args, "episode_days")
        self.episode_len = getattr(self.args, "episode_len")
        self.state_space = getattr(self.args, "state_space", ["peak", "power", "soc", "soh", "dod"])
        self.reward_type = getattr(self.args, "reward_type")
        self.socs_of_one_action = getattr(self.args, "socs_of_one_action")
        self.n_actions = getattr(self.args, "n_actions") * self.n_agents // self.socs_of_one_action + 1
        self.index_to_action = getattr(self.args,
                                       "index_to_action") * self.n_agents // self.socs_of_one_action
        self.powers_data = getattr(self.args, "powers_data")
        self.powers = self.load_data()
        self.powers_sum = self.powers.sum(axis=0)
        self.peak0 = round(np.mean(self.powers_sum), 2)

        self.bess_list = list()
        for i in range(self.n_agents):
            self.bess_list.append(OneBESS(str(i), self.args))

        state = self.reset()
        self.state_size = state.shape[0]

    def load_data(self):
        if "-" in self.powers_data:
            start, end = list(map(int, self.powers_data.split("-", 1)))
            power_set = range(start, end)
        elif "_" in self.powers_data:
            power_set = self.powers_data.split("_")
        else:
            power_set = [self.powers_data]
        if len(power_set) != self.n_agents:
            raise RuntimeError
        powers_data_path = os.path.join(os.path.dirname(__file__), "data/power_data")
        files = [os.path.join(powers_data_path, "EC_{}.txt".format(i)) for i in
                 power_set]
        powers = np.array([np.loadtxt(f) for f in files])
        return powers

    def __str__(self):
        return self.map_name

    def reset(self):
        self.time = 0
        self.peak = self.peak0
        self.powers_t = self.powers[:, self.time]
        for i in range(self.n_agents):
            self.bess_list[i].reset()

        return self.get_state()

    def step(self, action):
        delta_soc_int = self._index_to_action(action) * self.socs_of_one_action
        if self.n_agents != 1:
            actions = self.allocation_method(delta_soc_int)
        else:
            actions = np.array([delta_soc_int])
        actions = actions / 100

        ec_powers_after = []
        delta_SoHs = []
        for i in range(self.n_agents):
            action = actions[i]
            ec_delta_power = action * self.bess_list[i].E_c * self.bess_list[i].SoH * self.get_efficiency(i,
                                                                                                          action) / self.delta_t
            delta_SoH = self.bess_list[i].do_action(action)
            delta_SoHs.append(delta_SoH)
            ec_power_after = round(self.powers_t[i] - ec_delta_power, 3)
            if ec_power_after < 0:
                ec_power_after = 0
            ec_powers_after.append(ec_power_after)

        sum_powers_after = sum(ec_powers_after)
        peak_charge = max(0, (sum_powers_after - self.peak)) * self.p_d * self.episode_days
        self.peak = max(sum_powers_after, self.peak)
        energy_charge = sum_powers_after * self.p_e * self.delta_t
        battery_cost = sum(
            [delta_SoHs[i] * self.bess_list[i].bat_price * self.bess_list[i].E_c for i in range(self.n_agents)])
        if self.time < self.episode_len - 1:
            terminated = False
        else:
            terminated = True
        if self.reward_type == 0:
            reward = -(peak_charge + energy_charge + battery_cost)
        else:
            raise Exception("undefined reward type")
        self.time += 1
        if not terminated:
            self.powers_t = self.powers[:, self.time]
        info = dict()
        info["p_after"] = ec_powers_after
        return reward, terminated, info

    def allocation_method(self, delta_soc_int):
        socs = np.array([self.bess_list[i].SoC for i in range(self.n_agents)])
        solution = np.zeros(self.n_agents)
        if delta_soc_int == 0:
            return np.zeros(self.n_agents)
        elif delta_soc_int > 0:
            socs_from_P_max_D = np.array([(self.bess_list[i].P_max_D * self.delta_t / self.bess_list[i].eta_D) / (
                    self.bess_list[i].E_c * self.bess_list[i].SoH) for i in range(self.n_agents)])
            socs_from_powers = np.array([self.powers[i, self.time] * self.delta_t / (
                    self.bess_list[i].E_c * self.bess_list[i].SoH * self.bess_list[i].eta_D) for i in
                                         range(self.n_agents)])
            socs_from_soc_min = socs - np.array([self.bess_list[i].min_soc for i in range(self.n_agents)])
            socs_avail_perc = np.stack((socs_from_soc_min, socs_from_P_max_D, socs_from_powers), axis=1)
            socs_avail_perc = np.where(socs_avail_perc > 0, socs_avail_perc, 0).min(axis=1)
            socs_avail = np.floor(socs_avail_perc * 100)
            if socs_avail.sum() < delta_soc_int:
                solution = socs_avail
            else:
                for i in range(1, delta_soc_int + 1):
                    max_soc_index = np.argmax(socs_avail)
                    socs_avail[max_soc_index] -= 1
                    solution[max_soc_index] += 1

        else:
            socs_from_P_max_C = np.array([(self.bess_list[i].P_max_C * self.delta_t / self.bess_list[i].eta_C) / (
                    self.bess_list[i].E_c * self.bess_list[i].SoH) for i in range(self.n_agents)])
            socs_from_soc_max = np.array([self.bess_list[i].max_soc for i in range(self.n_agents)]) - socs
            socs_avail_perc = np.stack((socs_from_P_max_C, socs_from_soc_max), axis=1)
            socs_avail_perc = np.where(socs_avail_perc > 0, socs_avail_perc, 0).min(axis=1)
            socs_avail = np.round(socs_avail_perc * 100)
            if socs_avail.sum() < abs(delta_soc_int):
                solution = -socs_avail
            else:
                for i in range(1, abs(delta_soc_int) + 1):
                    max_soc_index = np.argmax(socs_avail)
                    socs_avail[max_soc_index] -= 1
                    solution[max_soc_index] += 1
                solution = -solution
        if sum(solution) != delta_soc_int:
            print("{}!={}".format(solution, delta_soc_int))
        return solution

    def get_state(self):
        state = list()
        state.append(self.peak)
        if "power" in self.state_space:
            state += [self.powers_t[i] for i in range(self.n_agents)]
        if "soc" in self.state_space:
            state += [self.bess_list[i].SoC for i in range(self.n_agents)]
        if "soh" in self.state_space:
            state += [self.bess_list[i].SoH for i in range(self.n_agents)]
        if "dod" in self.state_space:
            state += [self.bess_list[i].DoD for i in range(self.n_agents)]
        state = np.array(state)
        return state

    def get_state_size(self):
        return self.state_size

    def get_efficiency(self, i, action):
        if action > 0:
            return self.bess_list[i].eta_D
        else:
            return 1 / self.bess_list[i].eta_C

    def get_total_actions(self):
        return self.n_actions

    def get_avail_agent_actions(self):
        return self.get_avail_agent_actions_via_state(self.get_state())

    def get_avail_agent_actions_via_state(self, state):
        if "power" not in self.state_space:
            raise RuntimeError
        powers = state[1:1 + self.n_agents]
        if "soc" not in self.state_space:
            raise RuntimeError
        socs = state[1 + self.n_agents:1 + self.n_agents * 2]
        peak_power = self.peak
        if peak_power > sum(powers):
            max_delta_soc = 0
        else:
            delta_soc_from_P_max_D = [(self.bess_list[i].P_max_D * self.delta_t) / (
                    self.bess_list[i].E_c * self.bess_list[i].SoH) for i in range(self.n_agents)]
            delta_soc_from_cur_soc = [round(socs[i] - self.bess_list[i].min_soc, 2) for i in
                                      range(self.n_agents)]
            soc_peak_cur_power = clip_float_2((np.sum(powers) - peak_power) * self.delta_t / (
                np.mean([self.bess_list[i].E_c * self.bess_list[i].SoH * self.bess_list[i].eta_D for i in
                         range(self.n_agents)]))) + 0.01  # 向上取整

            max_soc_batch = self.get_min_batch((delta_soc_from_P_max_D, delta_soc_from_cur_soc))
            max_delta_soc = max(min(np.sum(max_soc_batch), soc_peak_cur_power), 0)

        delta_soc_from_P_max_C = [-(self.bess_list[i].P_max_C * self.delta_t) / (
                self.bess_list[i].E_c * self.bess_list[i].SoH) for
                                  i in range(self.n_agents)]
        delta_soc_from_cur_soc_min = [-round((self.bess_list[i].max_soc - socs[i]), 2) for i in
                                      range(self.n_agents)]

        min_soc_batch = self.get_max_batch((delta_soc_from_P_max_C, delta_soc_from_cur_soc_min))

        min_delta_soc = np.sum(min_soc_batch)
        min_action = round(min_delta_soc * 100 / self.socs_of_one_action)
        max_action = round(max_delta_soc * 100 / self.socs_of_one_action)

        avail_actions = np.zeros(self.n_actions)
        avail_actions[self._action_to_index(min_action): self._action_to_index(max_action) + 1] = 1
        return avail_actions

    def get_max_batch(self, l):
        s = np.stack(l, axis=1)
        s = np.where(s < 0, s, 0)
        m = s.max(axis=1)
        return m

    def get_min_batch(self, l):
        s = np.stack(l, axis=1)
        s = np.where(s > 0, s, 0)
        m = s.min(axis=1)
        return m

    def get_num_of_agents(self):
        return self.n_agents

    def _index_to_action(self, action_index):
        return action_index - self.index_to_action

    def _action_to_index(self, action):
        return action + self.index_to_action
