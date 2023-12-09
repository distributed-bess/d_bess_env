import xlrd
import yaml
from scipy import interpolate
from collections import namedtuple
import math
import numpy as np
import os


def read_excel(i, wb):
    sheet = wb.sheet_by_index(0)
    x = sheet.cell(i, 0).value
    y = sheet.cell(i, 1).value
    return x, y


def read_dod(bat_type, wb):
    dod_level = []
    number_of_cycles = []
    dod_file_len = {"LA": 27,
                    "LFP": 35}
    for x in range(0, dod_file_len[bat_type]):
        x, y = read_excel(x, wb)
        dod_level.append(x)
        number_of_cycles.append(y)
    min_dod_level = dod_level[0]
    max_number_of_cycles = number_of_cycles[0]
    func = interpolate.interp1d(dod_level, number_of_cycles, kind='cubic')
    return min_dod_level, max_number_of_cycles, func


def dod(bat_type):
    dod_files = {"LA": os.path.join(os.path.dirname(__file__), "data/dod_data/LA_dod.xlsx"),
                 "LFP": os.path.join(os.path.dirname(__file__), "data/dod_data/LFP_dod.xlsx")}
    wb = xlrd.open_workbook(filename=dod_files[bat_type])

    def dod_func(dod_level):
        min_dod_level, max_number_of_cycles, func = read_dod(bat_type, wb)
        if dod_level < min_dod_level / 2:
            return max_number_of_cycles * 4
        elif dod_level < min_dod_level:
            return max_number_of_cycles * 2
        elif dod_level > 100:
            dod_level = 100
        return func(dod_level)

    return dod_func


def float_equal(a, b):
    if math.fabs(a - b) < 1e-10:
        return True
    else:
        return False


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def load_env_args(map_name):
    env_config_file_path = os.path.join(os.path.dirname(__file__),
                                        "config/{}.yaml".format(map_name))
    with open(env_config_file_path, "r") as f:
        env_config = yaml.safe_load(f)
        env_config_dict = env_config["env_args"]
        env_config_dict["episode_len"] = int(env_config_dict["episode_days"] * 24 / env_config_dict["delta_t"])
        bat_config_dict = env_config[env_config_dict["bat_type"]]
        env_config_dict.update(bat_config_dict)
        P_R = env_config_dict["P_R"]
        env_config_dict["max_charge_power"] = P_R * env_config_dict["charge_efficiency"] / 5
        env_config_dict["max_discharge_power"] = P_R / env_config_dict["discharge_efficiency"]
        args = env_config_dict
    return convert(args)


class OneBESS:
    def __init__(self, name, kwargs):
        self.name = name
        self.args = kwargs
        self.E_c = getattr(self.args, "bat_capacity")
        self.P_max_D = getattr(self.args, "max_discharge_power")
        self.P_max_C = getattr(self.args, "max_charge_power")
        self.eta_D = getattr(self.args, "discharge_efficiency")
        self.eta_C = getattr(self.args, "charge_efficiency")
        self.min_soc = getattr(self.args, "min_soc")
        self.max_soc = getattr(self.args, "max_soc")
        self.SoH_dead = getattr(self.args, "soh_dead")
        self.dod_func = dod(getattr(self.args, "bat_type"))
        self.episode_len = getattr(self.args, "episode_len")
        self.delta_t = getattr(self.args, "delta_t")
        self.bat_price = getattr(self.args, "bat_price")
        self.init_soh = getattr(self.args, "init_soh")

        self.reset()

    def __str__(self):
        return self.name

    def reset(self):
        self.SoH = self.init_soh
        self.SoC = getattr(self.args, "init_soc")
        self.DoD = getattr(self.args, "init_dod")

    def do_action(self, action, time_slot):
        delta_SoC = action
        delta_SoH = 0
        if not float_equal(action, 0):
            if action > 0:
                self.SoC -= delta_SoC
                delta_SoH_1 = (1 - self.SoH_dead) / self.dod_func(self.DoD * 100)
                self.DoD += delta_SoC
                delta_SoH_2 = (1 - self.SoH_dead) / self.dod_func(self.DoD * 100)
                delta_SoH = delta_SoH_2 - delta_SoH_1
                self.SoH -= delta_SoH
            else:
                self.SoC -= delta_SoC
                self.SoC = round(self.SoC, 2)
                self.DoD = max(self.DoD + delta_SoC, 0)
                self.DoD = round(self.DoD, 2)
        self.SoC = np.clip(self.SoC, self.min_soc, self.max_soc)
        return delta_SoH

