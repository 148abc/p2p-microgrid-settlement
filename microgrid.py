import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def get_grid_price(time_slots=24):
    """"""
    price = np.zeros(time_slots)
    for t in range(time_slots):
        if 10 <= t < 15 or 18 <= t < 21:
            price[t] = 0.20
        elif 7 <= t < 10 or 15 <= t < 18:
            price[t] = 0.10
        else:
            price[t] = 0.045
    return price


def get_fit_price(time_slots=24):
    """"""
    price = np.zeros(time_slots)
    for t in range(time_slots):
        if 6 <= t < 22:
            price[t] = 0.036
    return price


class Microgrid:
    """"""
    def __init__(self, mg_id, mg_type, time_slots=24, real_data=None, soc_bounds=None):
        """"""
        self.mg_id = mg_id
        self.time_slots = time_slots

        if real_data is not None:
            self.mg_type = real_data["mg_type"]
            self.load = np.asarray(real_data["load"], dtype=float)
            self.pv_generation = np.asarray(real_data["pv"], dtype=float)
            self.wind_generation = np.asarray(real_data["wind"], dtype=float)
            params = real_data["params"]
            self.max_load = params["max_load"]
            self.pv_capacity = params["pv_capacity"]
            self.wind_capacity = params["wind_capacity"]
            self.has_diesel = params.get("has_diesel", False)
            self.diesel_capacity = params.get("diesel_capacity", 0.0)
            self.dg_cost = params.get("dg_cost")
            self.battery_cost = params.get("battery_cost", 0.0)
            self.battery_capacity = params["battery_capacity"]
            self.battery_power = params["battery_power"]
            self.battery_efficiency = params["battery_efficiency"]
            self.p_grid_max = max(1000.0, float(np.ceil(1.5 * self.load.max())))
        else:
            self.mg_type = mg_type
            self._set_parameters()

            self.load = self._generate_load()
            self.pv_generation = self._generate_pv()
            self.wind_generation = self._generate_wind()
            self.p_grid_max = 1000.0
            self.dg_cost = None
            self.battery_cost = 0.0

        self.total_generation = self.pv_generation + self.wind_generation
        self.net_load = self.load - self.total_generation

        self.grid_price = get_grid_price(self.time_slots)
        self.fit_price = get_fit_price(self.time_slots)

        if soc_bounds is not None:
            self.soc_lb = np.asarray(soc_bounds["lb"], dtype=float)
            self.soc_ub = np.asarray(soc_bounds["ub"], dtype=float)
        else:
            self.soc_lb = np.full(time_slots + 1, 0.1 * self.battery_capacity)
            self.soc_ub = np.full(time_slots + 1, 0.9 * self.battery_capacity)

        self.battery_soc, self.battery_power_profile = self._simulate_battery()

    def _set_parameters(self):
        """"""
        if self.mg_type == "industrial":
            self.max_load = 600
            self.pv_capacity = 150  # kW
            self.wind_capacity = 150
            self.has_diesel = True
            self.diesel_capacity = 300  # kW
            self.battery_capacity = 500  # kWh
            self.battery_power = 200  # kW
            self.battery_efficiency = 0.95

        elif self.mg_type == "commercial":
            self.max_load = 300  # kW
            self.pv_capacity = 350
            self.wind_capacity = 0  # kW
            self.has_diesel = False
            self.battery_capacity = 300  # kWh
            self.battery_power = 150  # kW
            self.battery_efficiency = 0.95

        elif self.mg_type == "residential":
            self.max_load = 200  # kW
            self.pv_capacity = 160
            self.wind_capacity = 0  # kW
            self.has_diesel = False
            self.battery_capacity = 600  # kWh
            self.battery_power = 200  # kW
            self.battery_efficiency = 0.95

    def _generate_load(self):
        """"""
        t = np.arange(self.time_slots)

        if self.mg_type == "industrial":
            load = self.max_load * (0.7 + 0.2 * np.sin(np.pi * (t - 6) / 12))
            load = np.clip(load, 0.6 * self.max_load, self.max_load)

        elif self.mg_type == "commercial":
            load = self.max_load * (
                0.3 + 0.6 * np.maximum(0, np.sin(np.pi * (t - 6) / 12))
            )

        elif self.mg_type == "residential":
            morning_peak = 0.4 * np.exp(-((t - 7) ** 2) / 8)
            evening_peak = 0.6 * np.exp(-((t - 19) ** 2) / 8)
            load = self.max_load * (0.3 + morning_peak + evening_peak)

        else:
            raise ValueError(f"未知的微网类型: {self.mg_type}")

        return load

    def _generate_pv(self):
        """"""
        t = np.arange(self.time_slots)
        pv = self.pv_capacity * np.maximum(0, np.sin(np.pi * (t - 6) / 12))
        return pv

    def _generate_wind(self):
        """"""
        if self.wind_capacity == 0:
            return np.zeros(self.time_slots)

        t = np.arange(self.time_slots)
        base_wind = 0.4 + 0.3 * np.cos(np.pi * (t - 3) / 12)
        rng = np.random.RandomState(42 + self.mg_id)
        fluctuation = 0.2 * rng.randn(self.time_slots)
        wind = self.wind_capacity * np.clip(base_wind + fluctuation, 0.1, 0.9)
        return wind

    def _simulate_battery(self):
        """"""
        soc = np.zeros(self.time_slots + 1)
        power = np.zeros(self.time_slots)

        soc_target = 0.5 * (self.soc_lb[0] + self.soc_ub[0])
        soc[0] = soc_target
        soc_min = self.soc_lb
        soc_max = self.soc_ub

        for t in range(self.time_slots):
            net = self.net_load[t]

            if t == self.time_slots - 1:
                soc_diff = soc[t] - soc_target
                if soc_diff > 0:
                    power[t] = min(
                        soc_diff * self.battery_efficiency, self.battery_power
                    )
                    soc[t + 1] = soc[t] - power[t] / self.battery_efficiency
                elif soc_diff < 0:
                    power[t] = -min(
                        -soc_diff / self.battery_efficiency, self.battery_power
                    )
                    soc[t + 1] = soc[t] - power[t] * self.battery_efficiency
                else:
                    soc[t + 1] = soc[t]
                continue

            if net < 0:
                charge_power = min(-net, self.battery_power)
                charge_energy = charge_power * self.battery_efficiency
                if soc[t] + charge_energy <= soc_max[t + 1]:
                    power[t] = -charge_power
                    soc[t + 1] = soc[t] + charge_energy
                else:
                    charge_energy = soc_max[t + 1] - soc[t]
                    power[t] = -charge_energy / self.battery_efficiency
                    soc[t + 1] = soc_max[t + 1]

            elif net > 0:
                discharge_power = min(net, self.battery_power)
                discharge_energy = discharge_power / self.battery_efficiency
                if soc[t] - discharge_energy >= soc_min[t + 1]:
                    power[t] = discharge_power
                    soc[t + 1] = soc[t] - discharge_energy
                else:
                    discharge_energy = soc[t] - soc_min[t + 1]
                    power[t] = discharge_energy * self.battery_efficiency
                    soc[t + 1] = soc_min[t + 1]

            else:
                soc[t + 1] = soc[t]

        return soc[:-1], power

    def get_info(self):
        """"""
        info = {
            "微网编号": self.mg_id,
            "类型": self.mg_type,
            "最大负荷": f"{self.max_load} kW",
            "光伏容量": f"{self.pv_capacity} kW",
            "风电容量": f"{self.wind_capacity} kW",
            "储能容量": f"{self.battery_capacity} kWh",
            "储能功率": f"{self.battery_power} kW",
            "日总负荷": f"{self.load.sum():.1f} kWh",
            "日总发电": f"{self.total_generation.sum():.1f} kWh",
            "日净负荷": f"{self.net_load.sum():.1f} kWh",
        }
        return info

    def plot(self):
        """"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        t = np.arange(self.time_slots)

        ax1.plot(t, self.load, "r-", label="负荷", linewidth=2)
        ax1.plot(t, self.pv_generation, "g-", label="光伏发电", linewidth=2)
        if self.wind_capacity > 0:
            ax1.plot(t, self.wind_generation, "c-", label="风电", linewidth=2)
        ax1.plot(t, self.battery_power_profile, "m-", label="储能功率", linewidth=2)
        ax1.plot(t, self.net_load, "b--", label="净负荷", linewidth=2)
        ax1.set_xlabel("时间 (h)")
        ax1.set_ylabel("功率 (kW)")
        ax1.set_title(f"微网 {self.mg_id} ({self.mg_type}) - 功率曲线")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

        ax2.plot(t, self.battery_soc, "orange", label="SOC", linewidth=2)
        ax2.axhline(
            y=0.9 * self.battery_capacity, color="r", linestyle="--", label="SOC上限"
        )
        ax2.axhline(
            y=0.1 * self.battery_capacity, color="r", linestyle="--", label="SOC下限"
        )
        ax2.set_xlabel("时间 (h)")
        ax2.set_ylabel("SOC (kWh)")
        ax2.set_title(f"微网 {self.mg_id} - 储能SOC")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class DistributionNetwork:
    """"""
    _LINE_PARAM_TEMPLATES = {
        (0, 1): {"R": 0.05, "X": 0.08, "length": 2.0, "capacity": 500},
        (1, 2): {"R": 0.06, "X": 0.10, "length": 2.5, "capacity": 400},
        (2, 3): {"R": 0.04, "X": 0.07, "length": 1.5, "capacity": 300},
    }

    def __init__(self, num_microgrids=3):
        """"""
        self.num_nodes = num_microgrids + 1
        self.num_microgrids = num_microgrids

        self.lines = [(i, i + 1) for i in range(num_microgrids)]

        default_param = {"R": 0.05, "X": 0.08, "length": 2.0, "capacity": 400}
        self.line_params = {}
        for line in self.lines:
            if line in self._LINE_PARAM_TEMPLATES:
                self.line_params[line] = self._LINE_PARAM_TEMPLATES[line].copy()
            else:
                self.line_params[line] = default_param.copy()

        self.line_impedance = {}
        for line, params in self.line_params.items():
            r_total = params["R"] * params["length"]
            x_total = params["X"] * params["length"]
            self.line_impedance[line] = (r_total, x_total)

    def get_line_capacity(self, from_node, to_node):
        """"""
        if (from_node, to_node) in self.line_params:
            return self.line_params[(from_node, to_node)]["capacity"]
        elif (to_node, from_node) in self.line_params:
            return self.line_params[(to_node, from_node)]["capacity"]
        else:
            return float("inf")

    def get_line_impedance(self, from_node, to_node):
        """"""
        if (from_node, to_node) in self.line_impedance:
            return self.line_impedance[(from_node, to_node)]
        elif (to_node, from_node) in self.line_impedance:
            return self.line_impedance[(to_node, from_node)]
        else:
            return (0, 0)

    def calculate_line_loss(self, power_flow, from_node, to_node):
        """"""
        R, X = self.get_line_impedance(from_node, to_node)
        V = 10  # kV
        current = abs(power_flow) / V
        loss = (current**2) * R / 1000
        return loss

    def get_network_info(self):
        """"""
        topo_str = (
            "PCC - "
            + " - ".join([f"MG{i + 1}" for i in range(self.num_microgrids)])
            + " (链式)"
        )
        info = {
            "节点数": self.num_nodes,
            "微网数": self.num_microgrids,
            "线路数": len(self.lines),
            "拓扑": topo_str,
            "线路容量(kW)": [self.get_line_capacity(f, t) for f, t in self.lines],
        }
        return info


def build_network(microgrids, margin=1.5, floor=500.0):
    """"""
    peaks = [float(np.max(mg.load)) for mg in microgrids]
    net = DistributionNetwork(len(microgrids))
    for k, line in enumerate(net.lines):
        net.line_params[line]["capacity"] = float(
            max(floor, margin * sum(peaks[k:]))
        )
    return net


if __name__ == "__main__":
    mg1 = Microgrid(1, "industrial")
    mg2 = Microgrid(2, "commercial")
    mg3 = Microgrid(3, "residential")

    print("=" * 50)
    print("\n分时电价:")
    grid_price = get_grid_price()
    print(f"  高峰时段 (10-15h, 18-21h): {0.40} $/kWh")
    print(f"  平段时段 (7-10h, 15-18h): {0.25} $/kWh")
    print(f"  低谷时段 (21-7h): {0.10} $/kWh")

    print("\n" + "=" * 50)
    for mg in [mg1, mg2, mg3]:
        print(f"\n微网 {mg.mg_id} 信息:")
        for key, value in mg.get_info().items():
            print(f"  {key}: {value}")
    print("\n" + "=" * 50)

    network = DistributionNetwork(3)
    print("\n配电网信息:")
    for key, value in network.get_network_info().items():
        print(f"  {key}: {value}")
    print("\n" + "=" * 50)

    for mg in [mg1, mg2, mg3]:
        mg.plot()

    plt.show()
