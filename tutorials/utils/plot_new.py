# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
import random

from ai_economist.foundation import landmarks, resources
from ai_economist.foundation.scenarios.utils import rewards, social_metrics

def plot_map(maps, locs, ax=None, cmap_order=None):
    world_size = np.array(maps.get("Wood")).shape
    max_health = {"Wood": 1, "Stone": 1, "House": 1}
    n_agents = len(locs)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        ax.cla()
    tmp = np.zeros((3, world_size[0], world_size[1]))
    cmap = plt.get_cmap("jet", n_agents)

    if cmap_order is None:
        cmap_order = list(range(n_agents))
    else:
        cmap_order = list(cmap_order)
        assert len(cmap_order) == n_agents

    scenario_entities = [k for k in maps.keys() if "source" not in k.lower()]
    for entity in scenario_entities:
        if entity == "House":
            continue
        elif resources.has(entity):
            if resources.get(entity).collectible:
                map_ = (
                    resources.get(entity).color[:, None, None]
                    * np.array(maps.get(entity))[None]
                )
                map_ /= max_health[entity]
                tmp += map_
        elif landmarks.has(entity):
            map_ = (
                landmarks.get(entity).color[:, None, None]
                * np.array(maps.get(entity))[None]
            )
            tmp += map_
        else:
            continue

    if isinstance(maps, dict):
        house_idx = np.array(maps.get("House")["owner"])
        house_health = np.array(maps.get("House")["health"])
    else:
        house_idx = maps.get("House", owner=True)
        house_health = maps.get("House")
    for i in range(n_agents):
        houses = house_health * (house_idx == cmap_order[i])
        agent = np.zeros_like(houses)
        agent += houses
        col = np.array(cmap(i)[:3])
        map_ = col[:, None, None] * agent[None]
        tmp += map_

    tmp *= 0.7
    tmp += 0.3

    tmp = np.transpose(tmp, [1, 2, 0])
    tmp = np.minimum(tmp, 1.0)

    ax.imshow(tmp, vmax=1.0, aspect="auto")

    bbox = ax.get_window_extent()

    for i in range(n_agents):
        r, c = locs[cmap_order[i]]
        col = np.array(cmap(i)[:3])
        ax.plot(c, r, "o", markersize=bbox.height * 20 / 550, color="w")
        ax.plot(c, r, "*", markersize=bbox.height * 15 / 550, color=col)

    ax.set_xticks([])
    ax.set_yticks([])


def plot_env_state(env, ax=None, remap_key=None):
    maps = env.world.maps
    locs = [agent.loc for agent in env.world.agents]

    if remap_key is None:
        cmap_order = None
    else:
        assert isinstance(remap_key, str)
        cmap_order = np.argsort(
            [agent.state[remap_key] for agent in env.world.agents]
        ).tolist()

    plot_map(maps, locs, ax, cmap_order)


def plot_log_state(dense_log, t, ax=None, remap_key=None):
    maps = dense_log["world"][t]
    states = dense_log["states"][t]

    n_agents = len(states) - 1
    locs = []
    for i in range(n_agents):
        r, c = states[str(i)]["loc"]
        locs.append([r, c])

    if remap_key is None:
        cmap_order = None
    else:
        assert isinstance(remap_key, str)
        key_val = np.array(
            [dense_log["states"][0][str(i)][remap_key] for i in range(n_agents)]
        )
        cmap_order = np.argsort(key_val).tolist()

    plot_map(maps, locs, ax, cmap_order)


def _format_logs_and_eps(dense_logs, eps):
    if isinstance(dense_logs, dict):
        return [dense_logs], [0]
    else:
        assert isinstance(dense_logs, (list, tuple))

    if isinstance(eps, (list, tuple)):
        return dense_logs, list(eps)
    elif isinstance(eps, (int, float)):
        return dense_logs, [int(eps)]
    elif eps is None:
        return dense_logs, list(range(np.minimum(len(dense_logs), 16)))
    else:
        raise NotImplementedError


def vis_world_array(dense_logs, ts, eps=None, axes=None, remap_key=None):
    dense_logs, eps = _format_logs_and_eps(dense_logs, eps)
    if isinstance(ts, (int, float)):
        ts = [ts]

    if axes is None:
        fig, axes = plt.subplots(
            len(eps),
            len(ts),
            figsize=(np.minimum(3.2 * len(ts), 16), 3 * len(eps)),
            squeeze=False,
        )

    else:
        fig = None

        if len(ts) == 1 and len(eps) == 1:
            axes = np.array([[axes]]).reshape(1, 1)
        else:
            try:
                axes = np.array(axes).reshape(len(eps), len(ts))
            except ValueError:
                print("Could not reshape provided axes array into the necessary shape!")
                raise

    for ti, t in enumerate(ts):
        for ei, ep in enumerate(eps):
            plot_log_state(dense_logs[ep], t, ax=axes[ei, ti], remap_key=remap_key)

    for ax, t in zip(axes[0], ts):
        ax.set_title("T = {}".format(t))
    for ax, ep in zip(axes[:, 0], eps):
        ax.set_ylabel("Episode {}".format(ep))

    return fig


def vis_world_range(
    dense_logs, t0=0, tN=None, N=5, eps=None, axes=None, remap_key=None
):
    dense_logs, eps = _format_logs_and_eps(dense_logs, eps)

    viable_ts = np.array([i for i, w in enumerate(dense_logs[0]["world"]) if w])
    if tN is None:
        tN = viable_ts[-1]
    assert 0 <= t0 < tN
    target_ts = np.linspace(t0, tN, N).astype(np.int)

    ts = set()
    for tt in target_ts:
        closest = np.argmin(np.abs(tt - viable_ts))
        ts.add(viable_ts[closest])
    ts = sorted(list(ts))
    if axes is not None:
        axes = axes[: len(ts)]
    return vis_world_array(dense_logs, ts, axes=axes, eps=eps, remap_key=remap_key)


def vis_builds(dense_logs, eps=None, ax=None):
    dense_logs, eps = _format_logs_and_eps(dense_logs, eps)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 3))
    cmap = plt.get_cmap("jet", len(eps))
    for i, ep in enumerate(eps):
        ax.plot(
            np.cumsum([len(b["builds"]) for b in dense_logs[ep]["Build"]]),
            color=cmap(i),
            label="Ep {}".format(ep),
        )
    ax.legend()
    ax.grid(b=True)
    ax.set_ylim(bottom=0)


def trade_str(c_trades, resource, agent, income=True):
    if income:
        p = [x["income"] for x in c_trades[resource] if x["seller"] == agent]
    else:
        p = [x["cost"] for x in c_trades[resource] if x["buyer"] == agent]
    if len(p) > 0:
        return "{:6.2f} (n={:3d})".format(np.mean(p), len(p))
    else:
        tmp = "~" * 8
        tmp = (" ") * 3 + tmp + (" ") * 3
        return tmp


def full_trade_str(c_trades, resource, a_indices, income=True):
    s_head = "{} ({})".format("Income" if income else "Cost", resource)
    ac_strings = [trade_str(c_trades, resource, buyer, income) for buyer in a_indices]
    s_tail = " | ".join(ac_strings)
    return "{:<15}: {}".format(s_head, s_tail)


def build_str(all_builds, agent):
    p = [x["income"] for x in all_builds if x["builder"] == agent]
    if len(p) > 0:
        return "{:6.2f} (n={:3d})".format(np.mean(p), len(p))
    else:
        tmp = "~" * 8
        tmp = (" ") * 3 + tmp + (" ") * 3
        return tmp


def full_build_str(all_builds, a_indices):
    s_head = "Income (Build)"
    ac_strings = [build_str(all_builds, builder) for builder in a_indices]
    s_tail = " | ".join(ac_strings)
    return "{:<15}: {}".format(s_head, s_tail)


def header_str(n_agents):
    s_head = ("_" * 15) + ":_"
    s_tail = "_|_".join([" Agent {:2d} ____".format(i) for i in range(n_agents)])
    return s_head + s_tail


def report(c_trades, all_builds, n_agents, a_indices=None):
    if a_indices is None:
        a_indices = list(range(n_agents))
    print(header_str(n_agents))
    resources = ["Wood", "Stone"]
    if c_trades is not None:
        for resource in resources:
            print(full_trade_str(c_trades, resource, a_indices, income=False))
        print("")
        for resource in resources:
            print(full_trade_str(c_trades, resource, a_indices, income=True))
    print(full_build_str(all_builds, a_indices))


def breakdown(log, remap_key=None):
    fig0 = vis_world_range(log, remap_key=remap_key)

    n = len(list(log["states"][0].keys())) - 1
    trading_active = "Trade" in log

    if remap_key is None:
        aidx = list(range(n))
    else:
        assert isinstance(remap_key, str)
        key_vals = np.array([log["states"][0][str(i)][remap_key] for i in range(n)])
        aidx = np.argsort(key_vals).tolist()

    all_builds = []
    for t, builds in enumerate(log["Build"]):
        if isinstance(builds, dict):
            builds_ = builds["builds"]
        else:
            builds_ = builds
        for build in builds_:
            this_build = {"t": t}
            this_build.update(build)
            all_builds.append(this_build)

    if trading_active:
        c_trades = {"Stone": [], "Wood": []}
        for t, trades in enumerate(log["Trade"]):
            if isinstance(trades, dict):
                trades_ = trades["trades"]
            else:
                trades_ = trades
            for trade in trades_:
                this_trade = {
                    "t": t,
                    "t_ask": t - trade["ask_lifetime"],
                    "t_bid": t - trade["bid_lifetime"],
                }
                this_trade.update(trade)
                c_trades[trade["commodity"]].append(this_trade)

        incomes = {
            "Sell Stone": [
                sum([t["income"] for t in c_trades["Stone"] if t["seller"] == aidx[i]])
                for i in range(n)
            ],
            "Buy Stone": [
                sum([-t["price"] for t in c_trades["Stone"] if t["buyer"] == aidx[i]])
                for i in range(n)
            ],
            "Sell Wood": [
                sum([t["income"] for t in c_trades["Wood"] if t["seller"] == aidx[i]])
                for i in range(n)
            ],
            "Buy Wood": [
                sum([-t["price"] for t in c_trades["Wood"] if t["buyer"] == aidx[i]])
                for i in range(n)
            ],
            "Build": [
                sum([b["income"] for b in all_builds if b["builder"] == aidx[i]])
                for i in range(n)
            ],
        }

    else:
        c_trades = None
        incomes = {
            "Build": [
                sum([b["income"] for b in all_builds if b["builder"] == aidx[i]])
                for i in range(n)
            ],
        }

    incomes["Total"] = np.stack([v for v in incomes.values()]).sum(axis=0)

    endows = [
        int(
            log["states"][-1][str(aidx[i])]["inventory"]["Coin"]
            + log["states"][-1][str(aidx[i])]["escrow"]["Coin"]
        )
        for i in range(n)
    ]

    n_small = np.minimum(4, n)

    # report(c_trades, all_builds, n, aidx)

    cmap = plt.get_cmap("jet", n)
    rs = ["Wood", "Stone", "Coin"]

    tmp = np.array(log["world"][0]["Stone"])
    fig1, axes = plt.subplots(
        2 if trading_active else 1, n_small,
        figsize=(16, 8 if trading_active else 4),
        sharex="row", sharey="row", squeeze=False,
    )
    # plot trajectories of agents
    for i, ax in enumerate(axes[0]):
        rows = np.array([x[str(aidx[i])]["loc"][0] for x in log["states"]]) * -1
        cols = np.array([x[str(aidx[i])]["loc"][1] for x in log["states"]])
        ax.plot(cols[::20], rows[::20])
        ax.plot(cols[0], rows[0], "r*", markersize=15)
        ax.plot(cols[-1], rows[-1], "g*", markersize=15)
        ax.set_title("Agent {}".format(i))
        ax.set_xlim([-1, 1 + tmp.shape[1]])
        ax.set_ylim([-(1 + tmp.shape[0]), 1])

    # plot trading history green for wood, brown for stone
    if trading_active:
        for i, ax in enumerate(axes[1]):
            for r in ["Wood", "Stone"]:
                tmp = [
                    (s["t"], s["income"]) for s in c_trades[r] if s["seller"] == aidx[i]
                ]
                if tmp:
                    ts, prices = [np.array(x) for x in zip(*tmp)]
                    ax.plot(
                        np.stack([ts, ts]),
                        np.stack([np.zeros_like(prices), prices]),
                        color=resources.get(r).color,
                    )
                    ax.plot(
                        ts, prices, ".", color=resources.get(r).color, markersize=12
                    )

                tmp = [(s["t"], -s["cost"]) for s in c_trades[r] if s["buyer"] == aidx[i]]
                if tmp:
                    ts, prices = [np.array(x) for x in zip(*tmp)]
                    ax.plot(
                        np.stack([ts, ts]),
                        np.stack([np.zeros_like(prices), prices]),
                        color=resources.get(r).color,
                    )
                    ax.plot(
                        ts, prices, ".", color=resources.get(r).color, markersize=12
                    )
            ax.plot([-20, len(log["states"]) + 19], [0, 0], "w-")
            # ax.set_ylim([-10.2, 10.2]);
            ax.set_xlim([-20, len(log["states"]) + 19])
            # ax.grid(b=True)
            ax.set_facecolor([0.3, 0.3, 0.3])


    # plot resources collection, income, reward
    fig2, axes = plt.subplots(2, 2, figsize=(16, 8), sharey=False)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # ==================== woods, stones, coins ====================
    for r, ax in zip(rs, [axes[0][0], axes[0][1], axes[1][0]]):
        for i in range(n):
            ax.plot([x[str(aidx[i])]["inventory"][r] + x[str(aidx[i])]["escrow"][r]
                    for x in log["states"]],
                label=i, color=cmap(i), lw=2
            )
        ax.set_title(r)
        ax.legend()
        # ax.grid(b=True)

    # ==================== rewards ====================
    ax = axes[1][1]
    for i in range(n):
        accum_reward = 0
        list_reward = []
        for x in log["rewards"]:
            accum_reward += x[str(aidx[i])]
            list_reward.append(accum_reward)
        ax.plot(list_reward, label=i, color=cmap(i), lw=2)
    accum_reward = 0
    list_reward = []
    for x in log["rewards"]:
        accum_reward += x['p']
        list_reward.append(accum_reward)
    ax.plot(list_reward, label='planner', color='pink', lw=2)
    ax.set_title("Rewards")
    ax.legend()
    # ax.grid(b=True)


    # plot labor distribution
    fig3, axes = plt.subplots(2, 2, figsize=(16, 8), sharey=False)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # ==================== labor ====================
    ax = axes[0][0]
    for i in range(n):
        ax.plot([x[str(aidx[i])]["endogenous"]["Labor"] for x in log["states"]],
            label=i, color=cmap(i), lw=2
        )
    ax.set_title("Labor")
    ax.legend()
    # ax.grid(b=True)

    # ==================== house built by agents ====================
    ax = axes[0][1]
    house_built = {}
    for i in range(n):
        house_built[i] = [0]
    for x in log["Build"]:
        for i in range(n):
            house_built[i].append(house_built[i][-1])
        for build_event in x:
            agent_idx = int(build_event['builder'])
            house_built[agent_idx][-1] += 1
    
    for i in range(n):
        ax.plot(house_built[i][1:], label=i, color=cmap(i), lw=2)
    ax.set_title("Houses Built")
    ax.legend()
    # ax.grid(b=True)

    
    # ==================== trees cut by agents ====================
    ax = axes[1][0]
    trees_cut = {}
    for i in range(n):
        trees_cut[i] = [0]
    # iterate for time step
    for x in log['Gather']:
        for i in range(n):
            trees_cut[i].append(trees_cut[i][-1])
        for gather_event in x:
            if gather_event['resource'] == 'Wood':
                agent_idx = int(gather_event['agent'])
                gather_num = int(gather_event['n'])
                trees_cut[agent_idx][-1] += gather_num

    for i in range(n):
        ax.plot(trees_cut[i][1:], label=i, color=cmap(i), lw=2)
    ax.set_title("Trees Cut")
    ax.legend()
    # ax.grid(b=True)

    # ==================== stone mined by agents ====================
    ax = axes[1][1]
    stone_mined = {}
    for i in range(n):
        stone_mined[i] = [0]
    # iterate for time step
    for x in log['Gather']:
        for i in range(n):
            stone_mined[i].append(stone_mined[i][-1])
        for gather_event in x:
            if gather_event['resource'] == 'Stone':
                agent_idx = int(gather_event['agent'])
                gather_num = int(gather_event['n'])
                stone_mined[agent_idx][-1] += gather_num

    for i in range(n):
        ax.plot(stone_mined[i][1:], label=i, color=cmap(i), lw=2)
    ax.set_title("Stone Mined")
    ax.legend()
    # ax.grid(b=True)

    
    


    # ==================== tax brackets ====================
    fig4, axes = plt.subplots(1, 1, figsize=(10, 6), sharey=False)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    ax = axes

    cutoff_list = []
    cutoff_flag = False
    rate_period = {}
    period_cnt = 0
    for x in log['PeriodicTax']:
        if isinstance(x, dict):
            if not cutoff_flag:
                for tax_cutoff_val in x['cutoffs']:
                    cutoff_list.append(int(tax_cutoff_val))
                cutoff_flag = True
            rate_list = []
            for tax_rate in x['schedule']:
                rate_list.append(tax_rate + period_cnt * 0.008)
            rate_period[period_cnt] = rate_list
            period_cnt += 1
    
    cutoffs_ext = cutoff_list + [cutoff_list[-1] + 50]
    num_bracket = len(cutoff_list)   

    # tab20
    cmap_tax = plt.get_cmap("tab20", period_cnt)
    for period in range(period_cnt):
        # ax.vlines(cutoff_list, -0.95, 1.0, colors='b', linestyles='dashed', lw=1)
        rate_list = rate_period[period]
        ax.hlines(rate_list, cutoff_list, cutoffs_ext[1:], lw=2, 
                colors=cmap_tax(period), label=f"Period {period}")

    ax.set_title("Tax Rate in Brackets")
    ax.set_xticks(cutoff_list[1:])
    ax.set_xlim([cutoffs_ext[0], cutoffs_ext[-1]])
    ax.set_ylim([-0.05, 1.0])
    ax.set_xlabel("Tax Bracket")
    ax.set_ylabel("Marginal Tax Rate(%)")
    ax.legend()
    # ax.grid(b=True)

    return (fig0, fig1, fig2, fig3, fig4), incomes, endows, c_trades, all_builds


def plot_for_each_n(y_fun, n, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    cmap = plt.get_cmap("jet", n)
    for i in range(n):
        ax.plot(y_fun(i), color=cmap(i), label=i)
    ax.legend()
    ax.grid(b=True)