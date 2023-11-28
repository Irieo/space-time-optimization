# SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown
#
# SPDX-License-Identifier: MIT

import pypsa
import pandas as pd
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mc
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker

matplotlib.use("Agg")

from solve_network import palette
from pypsa.plot import add_legend_patches
import seaborn as sns


### Plotting time-series data


def heatmap(data, month, year, ax):
    data = df[(df["snapshot"].dt.month == month)]

    snapshot = data["snapshot"]
    day = data["snapshot"].dt.day
    value = data["iteration 1"]
    value = value.values.reshape(
        int(24 / scaling), len(day.unique()), order="F"
    )  # 8 clusters of 3h in each day

    xgrid = (
        np.arange(day.max() + 1) + 1
    )  # The inner + 1 increases the length, the outer + 1 ensures days start at 1, and not at 0
    ygrid = np.arange(int(24 / scaling) + 1)  # hours (sampled or not) + extra 1

    ax.pcolormesh(xgrid, ygrid, value, cmap=colormap, vmin=MIN, vmax=MAX)
    # Invert the vertical axis
    ax.set_ylim(int(24 / scaling), 0)
    # Set tick positions for both axes
    ax.yaxis.set_ticks([])  # [i for i in range(int(24/scaling))]
    ax.xaxis.set_ticks([])
    # Remove ticks by setting their length to 0
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

    # Remove all spines
    ax.set_frame_on(False)


def heatmap_utilization(data, month, year, ax, carrier):
    data = df[(df["snapshot"].dt.month == month)]

    snapshot = data["snapshot"]
    day = data["snapshot"].dt.day
    value = data[f"{carrier}"]
    value = value.values.reshape(
        int(24 / scaling), len(day.unique()), order="F"
    )  # 8 clusters of 3h in each day

    xgrid = (
        np.arange(day.max() + 1) + 1
    )  # The inner + 1 increases the length, the outer + 1 ensures days start at 1, and not at 0
    ygrid = np.arange(int(24 / scaling) + 1)  # hours (sampled or not) + extra 1

    ax.pcolormesh(xgrid, ygrid, value, cmap=colormap, vmin=MIN, vmax=MAX)
    # Invert the vertical axis
    ax.set_ylim(int(24 / scaling), 0)
    # Set tick positions for both axes
    ax.yaxis.set_ticks([])  # [i for i in range(int(24/scaling))]
    ax.xaxis.set_ticks([])
    # Remove ticks by setting their length to 0
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

    # Remove all spines
    ax.set_frame_on(False)


def plot_heatmap_cfe():
    # Here 1 row year/variable and 12 columns for month
    fig, axes = plt.subplots(1, 12, figsize=(14, 5), sharey=True)
    plt.tight_layout()

    for i, year in enumerate([2013]):
        for j, month in enumerate(range(1, 13)):
            # print(f'j: {j}, month: {month}')
            heatmap(df, month, year, axes[j])

    # Adjust margin and space between subplots (extra space is on the left for a label)
    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.9, hspace=0.08, wspace=0
    )  # wspace=0 stacks individual months together but easy to split

    # some room for the legend in the bottom
    fig.subplots_adjust(bottom=0.2)

    # Create a new axis to contain the color bar
    # Values are: (x coordinate of left border, y coordinate for bottom border, width, height)
    cbar_ax = fig.add_axes([0.3, 0.03, 0.4, 0.04])

    # Create a normalizer that goes from minimum to maximum value
    norm = mc.Normalize(0, 1)  # for CFE, otherwise (MIN, MAX)

    # Create the colorbar and set it to horizontal
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=colormap),
        cax=cbar_ax,  # Pass the new axis
        orientation="horizontal",
    )

    # Remove tick marks and set label
    cb.ax.xaxis.set_tick_params(size=0)
    cb.set_label("Hourly CFE score of electricity supply from grid", size=12)

    # add some figure labels and title
    fig.text(0.5, 0.15, "Days of year", ha="center", va="center", fontsize=14)
    fig.text(
        0.03,
        0.5,
        "Hours of a day",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=14,
    )
    fig.suptitle(f"Carbon Heat Map | {location[:2]}", fontsize=20, y=0.98)

    path = snakemake.output.plot.split("capacity.pdf")[0] + f"heatmaps"
    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(
        path + "/" + f"{flex}_GridCFE_{location}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def plot_heatmap_utilization(carrier):
    # Here 1 row year/variable and 12 columns for month
    fig, axes = plt.subplots(1, 12, figsize=(14, 5), sharey=True)
    plt.tight_layout()

    for i, year in enumerate([2013]):
        for j, month in enumerate(range(1, 13)):
            # print(f'j: {j}, month: {month}')
            heatmap_utilization(df, month, year, axes[j], carrier=carrier)

    # Adjust margin and space between subplots (extra space is on the left for a label)
    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.9, hspace=0.08, wspace=0.04
    )  # wspace=0 stacks individual months together but easy to split

    # some room for the legend in the bottom
    fig.subplots_adjust(bottom=0.2)

    # Create a new axis to contain the color bar
    # Values are: (x coordinate of left border, y coordinate for bottom border, width, height)
    cbar_ax = fig.add_axes([0.3, 0.03, 0.4, 0.04])

    # Create a normalizer that goes from minimum to maximum value
    norm = mc.Normalize(MIN, MAX)

    # Create the colorbar and set it to horizontal
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=colormap),
        cax=cbar_ax,  # Pass the new axis
        orientation="horizontal",
    )

    # Remove tick marks and set label
    cb.ax.xaxis.set_tick_params(size=0)
    cb.set_label(f"Hourly {carrier} [MW]", size=12)

    # add some figure labels and title
    fig.text(0.5, 0.15, "Days of year", ha="center", va="center", fontsize=14)
    fig.text(
        0.03,
        0.5,
        "Hours of a day",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=14,
    )
    fig.suptitle(f"Flexibility utilization | {node}", fontsize=20, y=0.98)

    path = snakemake.output.plot.split("capacity.pdf")[0] + f"heatmaps"
    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(
        path + "/" + f"{flex}_{carrier.replace(' ', '_')}_{node}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def retrieve_nb(n, node):
    """
    Retrieve nodal energy balance per hour
        -> lines and links are bidirectional AND their subsets are exclusive.
        -> links include fossil gens
    NB {-1} multiplier is a nodal balance sign
    """

    components = ["Generator", "Load", "StorageUnit", "Store", "Link", "Line"]
    nodal_balance = pd.DataFrame(index=n.snapshots)

    for i in components:
        if i == "Generator":
            node_generators = n.generators.query("bus==@node").index
            nodal_balance = nodal_balance.join(n.generators_t.p[node_generators])
        if i == "Load":
            node_loads = n.loads.query("bus==@node").index
            nodal_balance = nodal_balance.join(-1 * n.loads_t.p_set[node_loads])
        if i == "Link":
            node_export_links = n.links.query("bus0==@node").index
            node_import_links = n.links.query("bus1==@node").index
            nodal_balance = nodal_balance.join(-1 * n.links_t.p0[node_export_links])
            nodal_balance = nodal_balance.join(-1 * n.links_t.p1[node_import_links])
            ##################
        if i == "StorageUnit":
            # node_storage_units = n.storage_units.query('bus==@node').index
            # nodal_balance = nodal_balance.join(n.storage_units_t.p_dispatch[node_storage_units])
            # nodal_balance = nodal_balance.join(n.storage_units_t.p_store[node_storage_units])
            continue
        if i == "Line":
            continue
        if i == "Store":
            continue

    nodal_balance = nodal_balance.rename(columns=rename).groupby(level=0, axis=1).sum()

    # Custom groupby function
    def custom_groupby(column_name):
        if column_name.startswith("vcc"):
            return "spatial shift"
        return column_name

    # Apply custom groupby function
    nodal_balance = nodal_balance.groupby(custom_groupby, axis=1).sum()

    # revert nodal balance sign for display
    if "spatial shift" in nodal_balance.columns:
        nodal_balance["spatial shift"] = nodal_balance["spatial shift"] * -1
    if "temporal shift" in nodal_balance.columns:
        nodal_balance["temporal shift"] = nodal_balance["temporal shift"] * -1

    return nodal_balance


def plot_balances(n, node, start="2013-03-01 00:00:00", stop="2013-03-08 00:00:00"):
    fig, ax = plt.subplots()
    fig.set_size_inches((8, 4.5))

    tech_colors = snakemake.config["tech_colors"]
    df = retrieve_nb(n, node)

    # format time & set a range to display
    ldf = df.loc[start:stop, :]
    duration = (pd.to_datetime(stop) - pd.to_datetime(start)).days
    ldf.index = pd.to_datetime(ldf.index, format="%%Y-%m-%d %H:%M:%S").strftime(
        "%m.%d %H:%M"
    )

    # get colors
    for item in ldf.columns:
        if item not in tech_colors:
            print("Warning!", item, "not in config/tech_colors")

    # set logical order
    new_index = preferred_order_balances.intersection(ldf.columns).append(
        ldf.columns.difference(preferred_order_balances)
    )
    ldf = ldf.loc[:, new_index]

    ldf.plot(
        kind="bar",
        stacked=True,
        color=tech_colors,
        ax=ax,
        width=1,
        # edgecolor = "black", linewidth=0.01
    )

    # visually ensure net energy balance at the node
    net_balance = ldf.sum(axis=1)
    x = 0
    for i in range(len(net_balance)):
        ax.scatter(x=x, y=net_balance[i], color="black", marker="_")
        x += 1

    plt.xticks(rotation=90)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set(xlabel=None)
    majors = range(0, duration * 24 + 1, 12)
    ax.xaxis.set_major_locator(ticker.FixedLocator(majors))

    ax.set_ylabel("Nodal balance [MW*h/h]")

    add_legend_patches(
        ax,
        colors=[tech_colors[c] for c in ldf.columns],
        labels=ldf.columns,
        legend_kw=dict(
            bbox_to_anchor=(1, 1),
            loc="upper left",
            frameon=False,
        ),
    )

    fig.tight_layout()

    _start = ldf.index[0].split(" ")[0]
    _stop = ldf.index[-1].split(" ")[0]
    path = snakemake.output.plot.split("capacity.pdf")[0] + f"{_start}-{_stop}"
    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(path + "/" + f"{flex}_balance_{node}.pdf")


####################################################################################################
if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_summary", year="2030", zone="IE", palette="p3", policy="cfe100"
        )

    config = snakemake.config
    scaling = int(config["time_sampling"][0])  # temporal scaling -- 3/1 for 3H/1H

    # Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:]) / 100 if policy != "ref" else 0
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year

    datacenters = snakemake.config["ci"]["datacenters"]
    locations = list(datacenters.keys())
    names = list(datacenters.values())
    flexibilities = snakemake.config["ci"]["flexibility"]

    # techs for CFE hourly matching
    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]
    storage_charge_techs = palette(tech_palette)[2]
    storage_discharge_techs = palette(tech_palette)[3]

    # renaming technologies for plotting
    clean_chargers = [g for g in storage_charge_techs]
    clean_chargers = [item.replace(" ", "_") for item in clean_chargers]
    clean_dischargers = [g for g in storage_discharge_techs]
    clean_dischargers = [item.replace(" ", "_") for item in clean_dischargers]

    exp_generators = [
        "offwind-ac-%s" % year,
        "offwind-dc-%s" % year,
        "onwind-%s" % year,
        "solar-%s" % year,
    ]
    exp_links = ["OCGT-%s" % year]
    exp_chargers = ["battery charger-%s" % year, "H2 Electrolysis-%s" % year]
    exp_dischargers = ["battery discharger-%s" % year, "H2 Fuel Cell-%s" % year]

    exp_generators = [item.replace(" ", "_") for item in exp_generators]
    exp_chargers = [item.replace(" ", "_") for item in exp_chargers]
    exp_dischargers = [item.replace(" ", "_") for item in exp_dischargers]

    # Assign colors
    tech_colors = snakemake.config["tech_colors"]

    rename_ci_cost = pd.Series(
        {
            "onwind": "onshore wind",
            "solar": "solar",
            "grid": "grid imports",
            "revenue": "revenue",
            "battery_storage": "battery",
            "battery_inverter": "battery",
            "battery_discharger": "battery",
            "hydrogen_storage": "hydrogen storage",
            "hydrogen_electrolysis": "hydrogen storage",
            "hydrogen_fuel_cell": "hydrogen storage",
            "adv_geothermal": "advanced dispatchable",
            "allam_ccs": "NG-Allam",
        }
    )

    rename_ci_capacity = pd.Series(
        {
            "onwind": "onshore wind",
            "solar": "solar",
            "battery_discharger": "battery",
            "H2_Fuel_Cell": "hydrogen fuel cell",
            "H2_Electrolysis": "hydrogen electrolysis",
            "adv_geothermal": "advanced dispatchable",
            "allam_ccs": "NG-Allam",
        }
    )

    rename_curtailment = pd.Series(
        {
            "offwind": "offshore wind",
            "offwind-ac": "offshore wind",
            "offwind-dc": "offshore wind",
            "onwind": "onshore wind",
            "solar": "solar",
            "ror": "hydroelectricity",
            "hydro": "hydroelectricity",
        }
    )

    preferred_order = pd.Index(
        [
            "advanced dispatchable",
            "NG-Allam",
            "Gas OC",
            "offshore wind",
            "onshore wind",
            "solar",
            "battery",
            "hydrogen storage",
            "hydrogen electrolysis",
            "hydrogen fuel cell",
        ]
    )

    rename_scen = {
        "0": "0.\n",
        "5": "0.05\n",
        "10": "0.1\n",
        "20": "0.2\n",
        "40": "0.4\n",
    }

    rename = {}

    for name in names:
        temp = {
            f"{name} H2 Electrolysis": "hydrogen storage",
            f"{name} H2 Fuel Cell": "hydrogen storage",
            f"{name} battery charger": "battery storage",
            f"{name} battery discharger": "battery storage",
            f"{name} export": "grid",
            f"{name} import": "grid",
            f"{name} onwind": "wind",
            f"{name} solar": "solar",
            f"{name} load": "load",
            f"{name} adv_geothermal": "clean dispatchable",
            f"{name} allam_ccs": "NG-Allam",
            f"{name} DSM-delayout": "temporal shift",
            f"{name} DSM-delayin": "temporal shift",
        }
        rename = rename | temp

    preferred_order_balances = pd.Index(
        [
            "clean dispatchable",
            "NG-Allam",
            "solar",
            "wind",
            "load",
            "battery storage",
            "hydrogen storage",
            "grid",
            "spatial shift",
            "temporal shift",
        ]
    )

    rename_system_simple = {
        "offwind-ac-%s" % year: "offshore wind",
        "offwind-dc-%s" % year: "offshore wind",
        "onwind-%s" % year: "onshore wind",
        "solar-%s" % year: "solar",
        "OCGT-%s" % year: "Gas OC",
        "battery_discharger-%s" % year: "battery",
        "H2_Fuel_Cell-%s" % year: "hydrogen fuel cell",
        "H2_Electrolysis-%s" % year: "hydrogen electrolysis",
    }


# SUMMARY PLOTS

# %matplotlib inline


# TIME-SERIES DATA (per flexibility scenario)

if snakemake.config["plot_timeseries"] == True:
    flexibilities = snakemake.config["scenario"]["flexibility"]

    # CARBON INTENSITY HEATMAPS

    for flex in flexibilities[0]:
        grid_cfe = pd.read_csv(
            snakemake.input.grid_cfe.split("0.csv")[0] + f"/{flex}.csv",
            index_col=0,
            header=[0, 1],
        )

        colormap = (
            "BrBG"  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        )

        for location in locations:
            # location = locations[-1]
            df = grid_cfe[f"{location}"]
            df = df.reset_index().rename(columns={"index": "snapshot"})
            df["snapshot"] = pd.to_datetime(df["snapshot"])

            MIN = df["iteration 1"].min()  # case of CFE -> 0
            MAX = df["iteration 1"].max()  # case of CFE -> 1

            plot_heatmap_cfe()

    # SPATIAL & TEMPORAL FLEXIBILITY UTILIZATION HEATMAPS

    for flex in flexibilities:
        n = pypsa.Network(snakemake.input.networks.split("0.nc")[0] + f"/{flex}.nc")
        colormap = "coolwarm"

        if snakemake.config["ci"]["temporal_shifting"] == True:
            for node in names:
                # node = names[0]
                temporal_shift = retrieve_nb(n, node).get("temporal shift")
                if temporal_shift is not None:
                    df = temporal_shift
                else:
                    df = pd.Series(
                        0, index=retrieve_nb(n, node).index, name="temporal shift"
                    )
                df = df.reset_index().rename(columns={"index": "snapshot"})
                df["snapshot"] = pd.to_datetime(df["snapshot"])
                MIN = -int(
                    flex
                )  # df["temporal shift"].min() #for flex co-opt case, value can exceed flex threshold
                MAX = +int(
                    flex
                )  # df["temporal shift"].max() #for flex co-opt case, value can exceed flex threshold

                plot_heatmap_utilization(carrier="temporal shift")

        if snakemake.config["ci"]["spatial_shifting"] == True:
            for node in names:
                # node = names[0]
                spatial_shift = retrieve_nb(n, node).get("spatial shift")
                if spatial_shift is not None:
                    df = spatial_shift
                else:
                    df = pd.Series(
                        0, index=retrieve_nb(n, node).index, name="spatial shift"
                    )
                df = df.reset_index().rename(columns={"index": "snapshot"})
                df["snapshot"] = pd.to_datetime(df["snapshot"])
                MIN = -int(
                    flex
                )  # df["spatial shift"].min() #for flex co-opt case, value can exceed flex threshold
                MAX = +int(
                    flex
                )  # df["spatial shift"].max() #for flex co-opt case, value can exceed flex threshold

                plot_heatmap_utilization(carrier="spatial shift")

        # NODAL BALANCES

        for node in names:
            plot_balances(n, node, "2013-03-01 00:00:00", "2013-03-08 00:00:00")
            plot_balances(n, node, "2013-05-01 00:00:00", "2013-05-08 00:00:00")
            plot_balances(n, node, "2013-12-01 00:00:00", "2013-12-08 00:00:00")

        print(f"Nodal balance completed for {flex} scen")
