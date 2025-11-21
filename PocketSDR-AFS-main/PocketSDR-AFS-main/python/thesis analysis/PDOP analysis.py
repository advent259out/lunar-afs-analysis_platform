import pylupnt as pnt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于显示进度的库
import time as pytime

# 记录开始时间
script_start_time = pytime.time()

# ==============================================================================
# 步骤 1: 设置时间和接收机
# ==============================================================================
print("步骤 1: 正在设置仿真时间和接收机...")
t0 = pnt.gregorian2time(1984, 5, 30, 16, 44, 48.0)
dt_total_2yr = 1.6 * pnt.DAYS_YEAR * pnt.SECS_DAY
dt_step_2yr = 15 * pnt.SECS_MINUTE
dt_prop_2yr = 1 * pnt.SECS_MINUTE
tspan_2yr = np.arange(0, dt_total_2yr + dt_step_2yr, dt_step_2yr)
tfs_2yr = t0 + tspan_2yr
n_steps = len(tfs_2yr)
min_elevation = 10 * pnt.RAD
r_south_pole_me = pnt.lat_lon_alt2cart(np.array([-90 * pnt.RAD, 0, 0]), pnt.R_MOON)

print(f"  起始时刻: {pnt.time2gregorian_string(t0)} TAI")
print(f"  仿真时长: 2 年，步长: {dt_step_2yr / 60.0} 分钟")

# ==============================================================================
# 步骤 2: 定义8卫星星座
# ==============================================================================
print("\n步骤 2: 正在设置8卫星星座...")
n_sat = 8
coes0_op_list = [
    [6540.0, 0.6, 56.3 * pnt.RAD, 0.0 * pnt.RAD, 90.0 * pnt.RAD, 0.0 * pnt.RAD],  # PRN-01
    [6540.0, 0.6, 56.3 * pnt.RAD, 0.0 * pnt.RAD, 90.0 * pnt.RAD, 90.0 * pnt.RAD],  # PRN-02
    [6540.0, 0.6, 56.3 * pnt.RAD, 0.0 * pnt.RAD, 90.0 * pnt.RAD, 180.0 * pnt.RAD],  # PRN-03
    [6540.0, 0.6, 56.3 * pnt.RAD, 0.0 * pnt.RAD, 90.0 * pnt.RAD, -90.0 * pnt.RAD],  # PRN-04
    [6540.0, 0.6, 56.3 * pnt.RAD, 180.0 * pnt.RAD, 90.0 * pnt.RAD, 45.0 * pnt.RAD],  # PRN-05
    [6540.0, 0.6, 56.3 * pnt.RAD, 180.0 * pnt.RAD, 90.0 * pnt.RAD, 135.0 * pnt.RAD],  # PRN-06
    [6540.0, 0.6, 56.3 * pnt.RAD, 180.0 * pnt.RAD, 90.0 * pnt.RAD, -135.0 * pnt.RAD],  # PRN-07
    [6540.0, 0.6, 56.3 * pnt.RAD, 180.0 * pnt.RAD, 90.0 * pnt.RAD, -45.0 * pnt.RAD],  # PRN-08
]
coes0_op = np.array(coes0_op_list)
rvs0_ci = np.zeros((n_sat, 6))
for i in range(n_sat):
    rv0_op_i = pnt.classical2cart(coes0_op[i], pnt.GM_MOON)
    rvs0_ci[i] = pnt.convert_frame(t0, rv0_op_i, pnt.MOON_OP, pnt.MOON_CI)

# ==============================================================================
# 步骤 3: 定义高精度动力学模型
# ==============================================================================
print("\n步骤 3: 正在配置高精度动力学模型...")
dyn_nbody = pnt.NBodyDynamics(pnt.IntegratorType.RK4)
dyn_nbody.add_body(pnt.Body.Moon(7, 1))
dyn_nbody.add_body(pnt.Body.Earth())
dyn_nbody.add_body(pnt.Body.Sun())
dyn_nbody.set_frame(pnt.MOON_CI)
dyn_nbody.set_time_step(dt_prop_2yr)

# ==============================================================================
# 步骤 4: 传播全部8颗卫星的轨道 (!!! 高计算量 !!!)
# ==============================================================================
print("\n步骤 4: --- 开始传播8卫星轨道 (!! 警告: 此过程将极其漫长 !!) ---")
rvs_ci = np.zeros((n_sat, n_steps, 6))
for i in range(n_sat):
    print(f"\n--- 正在传播 PRN-{i + 1:02d} (共 {n_sat} 颗) ---")
    rvs_ci[i] = dyn_nbody.propagate(rvs0_ci[i], t0, tfs_2yr, progress=True)
print("\n--- 轨道传播完成 ---")

# ==============================================================================
# 步骤 5: 分析覆盖率并计算 PDOP
# ==============================================================================
print("\n步骤 5: 正在分析覆盖率并计算 PDOP...")
pdop_history = np.full(n_steps, np.inf)  # 初始化PDOP历史，默认为无穷大
sats_in_view_history = np.zeros(n_steps, dtype=int)

print("  (正在转换坐标系至 MOON_ME...)")
rs_me = np.zeros((n_sat, n_steps, 3))
for i in tqdm(range(n_sat), desc="转换坐标系", unit="sat"):
    rs_me[i] = pnt.convert_frame(
        tfs_2yr, rvs_ci[i], pnt.MOON_CI, pnt.MOON_ME, rotate_only=True
    )[..., :3]

print("  (正在计算可见性和 PDOP...)")
for i in tqdm(range(n_steps), desc="计算 PDOP", unit="step"):
    t = tfs_2yr[i]
    r_sats_me_t = rs_me[:, i, :]  # 当前时刻所有卫星在 ME 系下的位置

    # 计算方位角、仰角、距离
    az_el_range = pnt.cart2az_el_range(r_sats_me_t, r_south_pole_me)
    elevations = az_el_range[:, 1]
    azimuths = az_el_range[:, 0]

    # 筛选可见卫星
    visible_mask = elevations >= min_elevation
    n_visible = np.sum(visible_mask)
    sats_in_view_history[i] = n_visible

    # --- PDOP 计算 (至少需要3颗卫星) ---
    if n_visible < 3:
        pdop_history[i] = np.inf
    else:
        el_vis = elevations[visible_mask]
        az_vis = azimuths[visible_mask]

        # 构建几何矩阵 H (n_visible x 3), 用于纯位置解算 (East, North, Up)
        H = np.zeros((n_visible, 3))
        H[:, 0] = np.cos(el_vis) * np.sin(az_vis)  # East
        H[:, 1] = np.cos(el_vis) * np.cos(az_vis)  # North
        H[:, 2] = np.sin(el_vis)  # Up

        try:
            Q = np.linalg.inv(H.T @ H)  # 协方差矩阵
            pdop_history[i] = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])  # PDOP
        except np.linalg.LinAlgError:
            pdop_history[i] = np.inf  # 矩阵奇异

print("PDOP 计算完成。")

# ==============================================================================
# 步骤 6: 打印统计数据
# ==============================================================================
print("\n步骤 6: 正在计算覆盖率统计数据...")
four_fold_coverage = np.sum(sats_in_view_history >= 4) / n_steps * 100
pdop_good_percent = np.sum((pdop_history <= 6)) / n_steps * 100  # 以 PDOP <= 6 为例

print("\n\n--- 月球南极点星座性能统计 (2年分析) ---")
print(f"  >= 4 颗卫星可见 (四重覆盖): {four_fold_coverage:>7.3f} %")
print(f"  PDOP <= 6.0:             {pdop_good_percent:>7.3f} %")
print(f"  平均可见卫星数:          {np.mean(sats_in_view_history):.2f} 颗")

# ==============================================================================
# 步骤 7: 绘制 PDOP 变化图 (使用英文标题)
# ==============================================================================
print("\n步骤 7: 正在生成 PDOP 变化图...")
fig, ax = plt.subplots(figsize=(14, 7))

time_days = tspan_2yr / pnt.SECS_DAY
ax.plot(time_days, pdop_history, lw=0.5, label='PDOP (Position-only, 3-sat min)')

# --- (FIXED) English Titles and Labels ---
ax.set_title(f"8-Sat Constellation PDOP at Lunar South Pole (2 Years)", fontsize=16)
ax.set_xlabel(f"Days Past {pnt.time2gregorian_string(t0)} TAI", fontsize=12)
ax.set_ylabel("PDOP (Position Dilution of Precision)", fontsize=12)
# --- End of Fix ---

# PDOP值可能非常大(无穷大)，我们设置一个合理的Y轴上限以便观察
ax.set_ylim(0, 20)  # Y轴上限设为20，以便观察
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig("pdop_analysis_8sat_south_pole.png")
print(f"\n图表已保存为 'pdop_analysis_8sat_south_pole.png'")

script_end_time = pytime.time()
total_time = script_end_time - script_start_time
print(f"\n--- 任务完成 ---")
print(f"总耗时: {total_time / 60.0:.2f} 分钟。")

plt.show()