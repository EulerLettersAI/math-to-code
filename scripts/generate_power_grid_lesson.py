"""Generate the three executable notebooks for the power-grid lesson."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lessons" / "power-flow-state-estimation"


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text, label=None):
    source = dedent(text).strip()
    if label:
        source = f"#| label: {label}\n#| echo: true\n\n{source}"
    return nbf.v4.new_code_cell(source)


def frontmatter(title, description, filename):
    return md(f"""
    ---
    title: "{title}"
    description: "{description}"
    author: "EulerLettersAI"
    date: last-modified
    jupyter: python3
    format:
      html:
        code-fold: false
        code-links:
          - text: Download notebook
            icon: file-earmark-arrow-down
            href: {filename}?download=1
    execute:
      echo: true
      warning: false
    ---
    """)


SETUP = r"""
%matplotlib inline
import json
import logging
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from pandapower.plotting import simple_plot

logging.getLogger("pandapower").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*numba cannot be imported.*")
pd.options.display.float_format = "{:,.5f}".format
"""


SETUP_MARKDOWN = """
## Setup

This notebook requires Python 3.11 and the packages below. Run this command in a notebook cell once if they are not already installed:

```python
%pip install matplotlib==3.10.8 numpy==2.2.6 pandas==2.3.3 pandapower==3.2.1
```

Restart the kernel after installation, then run the notebook from top to bottom.
"""


GRID = r"""
def build_ieee_14_grid():
    # Load pandapower's built-in IEEE 14-bus benchmark without changing it.
    return pn.case14()


def draw_grid(net, title="The pandapower IEEE 14-bus grid"):
    # case14 includes geographic bus coordinates; simple_plot is a pandapower tool.
    ax = simple_plot(
        net, show_plot=False, bus_size=0.7, line_width=2.0,
        ext_grid_size=1.1, trafo_size=0.8, plot_loads=True,
        plot_gens=True, plot_sgens=True, load_size=0.8,
        gen_size=0.8, sgen_size=0.8,
    )
    for bus, row in net.bus.iterrows():
        point = json.loads(row.geo)["coordinates"]
        ax.annotate(f"Bus {bus + 1}", point, xytext=(7, 7),
                    textcoords="offset points", fontsize=10)
    ax.set_title(title)
    return ax
"""


MEASUREMENTS = r"""
def add_redundant_measurements(net, seed=7):
    # Create reproducibly noisy, redundant SCADA-style measurements.
    rng = np.random.default_rng(seed)
    sigma_v, sigma_power = 0.005, 0.8

    # One voltage-magnitude measurement at every bus.
    for bus in net.bus.index:
        value = net.res_bus.at[bus, "vm_pu"] + rng.normal(0, sigma_v)
        pp.create_measurement(net, "v", "bus", value, sigma_v, bus,
                              name=f"V{bus}")

    # Active and reactive injection measurements at every bus.
    for bus in net.bus.index:
        for kind, column in [("p", "p_mw"), ("q", "q_mvar")]:
            value = net.res_bus.at[bus, column] + rng.normal(0, sigma_power)
            pp.create_measurement(net, kind, "bus", value, sigma_power, bus,
                                  name=f"{kind.upper()}{bus}_inj")

    # Active and reactive flow at every line's from end.
    for line in net.line.index:
        for kind, column in [("p", "p_from_mw"), ("q", "q_from_mvar")]:
            value = net.res_line.at[line, column] + rng.normal(0, sigma_power)
            pp.create_measurement(net, kind, "line", value, sigma_power,
                                  line, side="from",
                                  name=f"{kind.upper()}_{net.line.at[line, 'name']}_from")
    # Active and reactive flow at every transformer's high-voltage end.
    for trafo in net.trafo.index:
        for kind, column in [("p", "p_hv_mw"), ("q", "q_hv_mvar")]:
            value = net.res_trafo.at[trafo, column] + rng.normal(0, sigma_power)
            pp.create_measurement(net, kind, "trafo", value, sigma_power,
                                  trafo, side="hv",
                                  name=f"{kind.upper()}_T{trafo}_hv")
    return net
"""


def notebook(cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


nb1 = notebook([
    frontmatter("Power-Flow Analysis", "Solve the built-in IEEE 14-bus benchmark with pandapower.", "1.0-power-flow-analysis.ipynb"),
    md(SETUP_MARKDOWN),
    code(SETUP, "setup"),
    md("""
    ## Learning objectives

    By the end of this notebook, you will be able to:

    - distinguish **power flow** from **state estimation**;
    - write the nonlinear AC nodal-balance equations;
    - identify slack, PV, and PQ bus specifications;
    - explain a Newton–Raphson solve; and
    - calculate bus voltages, line flows, losses, and loading with pandapower.
    """),
    code(GRID, "grid-builder"),
    code("""
    net = build_ieee_14_grid()
    draw_grid(net, "IEEE 14-bus grid used throughout the lesson")
    plt.show()
    """, "draw-grid"),
    md("""
    ## 1. Given quantities

    A power-flow calculation starts from **specified operating quantities** and a **known network model**. It does not start from noisy meter readings. For the IEEE 14-bus case, the given quantities fall into four groups:

    1. the slack-bus voltage magnitude and angle;
    2. each generator's active-power set point and voltage-magnitude set point;
    3. each demand's active and reactive power; and
    4. the topology and electrical parameters of every line, transformer, and shunt.

    The bus-type table makes the boundary between inputs and unknowns explicit. Notebook indices are zero-based, while `IEEE bus` below uses the familiar one-based numbering.
    """),
    code("""
    slack_buses = set(net.ext_grid.bus.astype(int))
    generator_buses = set(net.gen.bus.astype(int))
    bus_specification = []
    for bus in net.bus.index:
        if bus in slack_buses:
            bus_type, given, solved = "Slack", "|V| and angle", "P and Q"
        elif bus in generator_buses:
            bus_type, given, solved = "PV", "P and |V|", "angle and Q"
        else:
            bus_type, given, solved = "PQ", "P and Q", "|V| and angle"
        bus_specification.append({
            "IEEE bus": bus + 1,
            "type": bus_type,
            "nominal kV": net.bus.at[bus, "vn_kv"],
            "given to power flow": given,
            "calculated by power flow": solved,
        })
    pd.DataFrame(bus_specification).set_index("IEEE bus")
    """, "given-bus-types"),
    md("""
    ### Given operating set points

    The following are the numerical source and demand inputs. Generator reactive power is **not** specified at a PV bus; it is calculated to maintain the specified voltage, subject to reactive limits when those limits are enforced. The slack source's $P$ and $Q$ are also outputs.
    """),
    code("""
    external_grid_given = net.ext_grid[["bus", "vm_pu", "va_degree"]].copy()
    external_grid_given["bus"] += 1
    external_grid_given.rename(columns={"bus": "IEEE bus"})
    """, "given-slack"),
    code("""
    generators_given = net.gen[[
        "bus", "p_mw", "vm_pu", "min_q_mvar", "max_q_mvar"
    ]].copy()
    generators_given["bus"] += 1
    generators_given.rename(columns={"bus": "IEEE bus"})
    """, "given-generators"),
    code("""
    loads_given = net.load[["bus", "p_mw", "q_mvar"]].copy()
    loads_given["bus"] += 1
    loads_given.rename(columns={"bus": "IEEE bus"})
    """, "given-loads"),
    code("""
    shunts_given = net.shunt[["bus", "p_mw", "q_mvar"]].copy()
    shunts_given["bus"] += 1
    shunts_given.rename(columns={"bus": "IEEE bus"})
    """, "given-shunts"),
    md("""
    ### Given network model

    The branch endpoints define the topology. Line resistance, reactance, capacitance, length, and rating are known; transformer ratings, short-circuit parameters, and tap positions are known. Together with nominal bus voltages and the system base power `net.sn_mva`, these data determine the admittance matrix $Y$.
    """),
    code("""
    lines_given = net.line[[
        "from_bus", "to_bus", "length_km", "r_ohm_per_km",
        "x_ohm_per_km", "c_nf_per_km", "max_i_ka"
    ]].copy()
    lines_given[["from_bus", "to_bus"]] += 1
    lines_given.rename(columns={"from_bus": "from IEEE bus", "to_bus": "to IEEE bus"})
    """, "given-lines"),
    code("""
    transformers_given = net.trafo[[
        "hv_bus", "lv_bus", "sn_mva", "vk_percent", "vkr_percent",
        "tap_side", "tap_neutral", "tap_pos", "tap_step_percent"
    ]].copy()
    transformers_given[["hv_bus", "lv_bus"]] += 1
    transformers_given.rename(columns={"hv_bus": "HV IEEE bus", "lv_bus": "LV IEEE bus"})
    """, "given-transformers"),
    code("""
    print(f"System base power: {net.sn_mva:.1f} MVA")
    print(f"Known topology: {len(net.bus)} buses, {len(net.line)} lines, "
          f"and {len(net.trafo)} transformers")
    """, "given-system-base"),
    md("""
    ## 2. The engineering question

    We use `pandapower.networks.case14()`, pandapower's built-in version of the IEEE 14-bus benchmark. It contains 14 buses, 15 lines, 5 transformers, an external grid, generators, loads, and shunts. All branch parameters and device set points are known.

    **Question.** For this operating scenario, what are the voltage magnitude and phase angle at every bus? What active and reactive power flows through every line, what are the losses, and does any line exceed its thermal rating?

    This is a **power-flow (load-flow) problem**: inputs are a network model and assumed injections/set points. No meter readings are required.
    """),
    md(r"""
    ## 3. AC power-flow equations

    Write the bus-voltage state as $V_i=|V_i|e^{j\theta_i}$ and the bus-admittance matrix entry as $Y_{ij}=G_{ij}+jB_{ij}$. With $\theta_{ij}=\theta_i-\theta_j$, the calculated net injections are

    $$
    P_i=\sum_{j=1}^{n}|V_i||V_j|\left(G_{ij}\cos\theta_{ij}+B_{ij}\sin\theta_{ij}\right),
    $$

    $$
    Q_i=\sum_{j=1}^{n}|V_i||V_j|\left(G_{ij}\sin\theta_{ij}-B_{ij}\cos\theta_{ij}\right).
    $$

    The balance equations set these calculated injections equal to generation minus demand. Bus types determine which quantities are specified:

    | Bus type | Specified | Solved |
    |---|---|---|
    | Slack | $|V|,\theta$ | $P,Q$ |
    | PV (generator) | $P,|V|$ | $\theta,Q$ |
    | PQ (load) | $P,Q$ | $|V|,\theta$ |

    In this example the external grid is the slack, generator buses are PV buses, and demand buses are PQ buses.
    """),
    md(r"""
    ## 4. How Newton–Raphson solves it

    At iteration $k$, form the mismatch vector from specified and calculated injections,

    $$
    g(x^{(k)})=
    \begin{bmatrix}\Delta P\\\Delta Q\end{bmatrix},
    \qquad
    x=\begin{bmatrix}\theta_{\text{non-slack}}\\|V|_{\text{PQ}}\end{bmatrix}.
    $$

    Linearize with the Jacobian $J=\partial(P,Q)/\partial(\theta,|V|)$ and solve

    $$J(x^{(k)})\,\Delta x=g(x^{(k)}),\qquad x^{(k+1)}=x^{(k)}+\Delta x.$$

    Iteration stops when the largest mismatch is below a tolerance. This is a square root-finding problem: one assumed operating scenario ideally produces one physically relevant solution.
    """),
    code("""
    pp.runpp(net, algorithm="nr", calculate_voltage_angles=True,
             init="flat", tolerance_mva=1e-8, numba=False)
    print(f"Converged: {net.converged}")
    """, "solve-power-flow"),
    md("""
    ## 5. Bus states and line flows

    pandapower uses the **consumer convention** in result tables: positive bus $P$ and $Q$ mean net consumption; negative values mean net injection into the grid.
    """),
    code("""
    bus_results = net.res_bus[["vm_pu", "va_degree", "p_mw", "q_mvar"]].copy()
    bus_results.index = net.bus.name
    bus_results
    """, "bus-results"),
    code("""
    line_results = net.res_line[[
        "p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "pl_mw", "loading_percent"
    ]].copy()
    line_results.index = net.line.name
    line_results
    """, "line-results"),
    code("""
    trafo_results = net.res_trafo[[
        "p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar", "pl_mw", "loading_percent"
    ]].copy()
    trafo_results.index = [f"Transformer {i}" for i in net.trafo.index]
    trafo_results
    """, "transformer-results"),
    code("""
    summary = pd.Series({
        "minimum voltage (pu)": net.res_bus.vm_pu.min(),
        "total branch active loss (MW)": net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum(),
        "maximum branch loading (%)": max(net.res_line.loading_percent.max(), net.res_trafo.loading_percent.max()),
    })
    summary
    """, "summary"),
    md("""
    ## 6. Interpretation and the bridge to state estimation

    The solution predicts the complete electrical condition from specified loads, generation, and the network model. In a control room, however, the actual loads are not known perfectly and meters contain noise. **State estimation reverses the information flow:** it uses redundant, imperfect measurements to infer the most plausible bus voltages, then derives injections and line flows from those estimated states.

    ### Check your understanding

    1. Increase the first load by 20%. Which bus voltage changes most?
    2. Open line 3 with `net.line.at[3, "in_service"] = False`. Does the case converge, and which branch loading changes most?
    3. Why does the slack-grid active power differ from the net specified demand?
    """),
])


nb2 = notebook([
    frontmatter("Weighted Least-Squares State Estimation", "Estimate the same grid from redundant noisy measurements.", "2.0-wls-state-estimation.ipynb"),
    md(SETUP_MARKDOWN),
    code(SETUP + "\nfrom pandapower.estimation import chi2_analysis, estimate", "setup"),
    md("""
    ## Learning objectives

    You will formulate nonlinear weighted least squares (WLS), construct a redundant measurement set, estimate bus-voltage states with pandapower, calculate flows from the estimated state, and compare estimation with power flow.
    """),
    code(GRID, "grid-builder"), code(MEASUREMENTS, "measurement-builder"),
    code("""
    net = build_ieee_14_grid()
    draw_grid(net, "The same IEEE 14-bus grid—now observed by meters")
    plt.show()
    """, "draw-grid"),
    md("""
    ## 1. The engineering question

    The network topology and parameters are known, but the true bus voltages and injections are not directly available. A SCADA system supplies noisy voltage, injection, and line-flow measurements with known standard deviations.

    **Question.** What set of bus voltage magnitudes and phase angles best explains all measurements simultaneously? From that estimated state, what are the line flows and loadings? Are the residuals consistent with the stated meter accuracy?

    Unlike load flow, state estimation does **not** treat every load and generator injection as exact. Redundancy lets inconsistent measurements be reconciled and later supports bad-data detection.
    """),
    md(r"""
    ## 2. WLS mathematical model

    For $n$ buses, choose the reference angle and define the $2n-1$ state vector

    $$x=[\theta_2,\ldots,\theta_n,|V_1|,\ldots,|V_n|]^T.$$

    Each measurement is a nonlinear function of the state:

    $$z=h(x)+e,\qquad e\sim\mathcal N(0,R),\qquad
    R=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_m^2).$$

    WLS minimizes standardized squared residuals,

    $$\hat x=\arg\min_x J(x),\qquad
    J(x)=[z-h(x)]^T R^{-1}[z-h(x)].$$

    At iteration $k$, with $H=\partial h/\partial x$, solve the Gauss–Newton normal equations

    $$
    (H^TR^{-1}H)\Delta x=H^TR^{-1}[z-h(x^{(k)})],
    \qquad x^{(k+1)}=x^{(k)}+\Delta x.
    $$

    Smaller $\sigma_i$ gives measurement $i$ more weight. A solution requires the gain matrix $G=H^TR^{-1}H$ to be nonsingular: this is the local numerical observability condition.
    """),
    md("""
    ## 3. Create redundant measurements

    A simulation needs hidden “truth” to synthesize meters. We first run a power flow, sample measurement noise with a fixed seed, and then estimate from **only** the stored measurements. In a real control room the physical system supplies those meter values; the true state is unavailable.

    There are $2(14)-1=27$ voltage-state unknowns and 82 measurements: 14 voltage magnitudes, 28 bus injections, 30 line-end flows, and 10 transformer-end flows.
    """),
    code("""
    pp.runpp(net, calculate_voltage_angles=True, numba=False)
    truth_bus = net.res_bus[["vm_pu", "va_degree"]].copy()
    truth_line = net.res_line[["p_from_mw", "q_from_mvar", "loading_percent"]].copy()
    add_redundant_measurements(net, seed=7)

    print(f"State variables: {2 * len(net.bus) - 1}")
    print(f"Measurements:   {len(net.measurement)}")
    net.measurement[["name", "measurement_type", "element_type", "value", "std_dev"]].head(10)
    """, "create-measurements"),
    code("""
    measurement_counts = (
        net.measurement.groupby(["element_type", "measurement_type"])
        .size().rename("count").to_frame()
    )
    measurement_counts
    """, "measurement-counts"),
    md("""## 4. Estimate the state with pandapower WLS"""),
    code("""
    result = estimate(net, algorithm="wls", init="flat", tolerance=1e-7,
                      maximum_iterations=50)
    result
    """, "run-wls"),
    code("""
    state_comparison = pd.DataFrame({
        "true_vm_pu": truth_bus.vm_pu,
        "estimated_vm_pu": net.res_bus_est.vm_pu,
        "vm_error_pu": net.res_bus_est.vm_pu - truth_bus.vm_pu,
        "true_angle_deg": truth_bus.va_degree,
        "estimated_angle_deg": net.res_bus_est.va_degree,
        "angle_error_deg": net.res_bus_est.va_degree - truth_bus.va_degree,
    }, index=net.bus.name)
    state_comparison
    """, "compare-states"),
    md("""## 5. Calculate flows from the estimated state

    Once $\hat x$ is known, pandapower evaluates the network equations to obtain estimated injections and branch flows—even where no flow meter exists."""),
    code("""
    estimated_flows = net.res_line_est[[
        "p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "loading_percent"
    ]].copy()
    estimated_flows.index = net.line.name
    estimated_flows
    """, "estimated-flows"),
    code("""
    estimated_transformer_flows = net.res_trafo_est[[
        "p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar", "loading_percent"
    ]].copy()
    estimated_transformer_flows.index = [f"Transformer {i}" for i in net.trafo.index]
    estimated_transformer_flows
    """, "estimated-transformer-flows"),
    code("""
    flow_comparison = pd.DataFrame({
        "true_p_from_mw": truth_line.p_from_mw,
        "estimated_p_from_mw": net.res_line_est.p_from_mw,
        "error_mw": net.res_line_est.p_from_mw - truth_line.p_from_mw,
    }, index=net.line.name)
    flow_comparison
    """, "compare-flows"),
    md(r"""
    ## 6. Residual consistency

    The minimized WLS statistic is compared with a $\chi^2$ threshold with approximately $m-(2n-1)$ degrees of freedom. A rejection means at least one of the meter-error model, topology, or network parameters is questionable; it does not identify the culprit by itself.
    """),
    code("""
    bad_data_detected = chi2_analysis(net, init="flat", tolerance=1e-7,
                                      maximum_iterations=50)
    print(f"Chi-square test flags bad data: {bad_data_detected}")
    """, "chi-square-test"),
    md("""
    ## 7. Power flow versus state estimation

    | Feature | Power flow | State estimation |
    |---|---|---|
    | Main question | What state follows from specified injections/set points? | What state best explains measured data? |
    | Inputs | Network plus assumed exact operating data | Network plus noisy, redundant meters |
    | Equations | $g(x)=0$ | $\min [z-h(x)]^TR^{-1}[z-h(x)]$ |
    | Typical solver | Newton–Raphson root finding | Gauss–Newton WLS |
    | Redundancy | Not required | Essential for robustness and bad-data checks |
    | Output | Predicted operating point | Reconstructed operating point and residuals |

    ### Check your understanding

    1. Double every meter standard deviation. Does the WLS point estimate change? What changes statistically?
    2. Add 8 MW to one active-power meter. Does the $\chi^2$ test flag the data?
    3. Remove measurements until WLS fails. Is a raw count of 27 measurements sufficient to guarantee observability?
    """),
])


nb3 = notebook([
    frontmatter("Observability in State Estimation", "Use rank, topology, redundancy, and WLS experiments to understand observability.", "3.0-state-estimation-observability.ipynb"),
    md(SETUP_MARKDOWN),
    code(SETUP + "\nfrom pandapower.estimation import estimate", "setup"),
    md("""
    ## Learning objectives

    You will distinguish topological from numerical observability, test Jacobian rank, see why measurement location matters more than count, connect observability to a spanning tree, and observe WLS failure when the gain matrix is singular.
    """),
    code(GRID, "grid-builder"), code(MEASUREMENTS, "measurement-builder"),
    code("""
    net = build_ieee_14_grid()
    draw_grid(net, "Observability of the IEEE 14-bus grid depends on meter placement")
    plt.show()
    """, "draw-grid"),
    md("""
    ## 1. The engineering question

    Suppose communication failures remove some meters from the same IEEE 14-bus grid.

    **Question.** Can the remaining measurements uniquely determine all bus-voltage magnitudes and non-reference angles? Which placements leave an unobservable island, how does redundancy improve resilience, and what symptom appears in a WLS estimator when observability is lost?
    """),
    md(r"""
    ## 2. Local numerical observability

    Linearize $z=h(x)+e$ near an operating point:

    $$\Delta z\approx H\Delta x,
    \qquad H=\left.\frac{\partial h}{\partial x}\right|_{x_0}.$$

    With one reference angle removed, the AC state has $n_x=2n-1$ entries. It is locally observable if

    $$\operatorname{rank}(H)=n_x.$$

    Equivalently, for positive measurement weights, the gain matrix

    $$G=H^TR^{-1}H$$

    must be nonsingular. Merely having $m\ge n_x$ measurements is **necessary but not sufficient**: repeated measurements in one area cannot reveal an unmeasured island.
    """),
    md(r"""
    ## 3. A transparent DC observability model

    The full AC Jacobian contains both angle and magnitude sensitivities. To make placement visible, first use the DC approximation: $|V|\approx1$, resistance is neglected, and line $\ell=(i,j)$ has

    $$P_{ij}\approx b_{ij}(\theta_i-\theta_j).$$

    Set $\theta_0=0$. Each measured branch flow contributes a row to a reduced matrix $H_{\mathrm{DC}}$. The 13 unknown angles are observable exactly when that matrix has rank 13. Both transmission lines and transformers are branches. This angle-only demonstration is not a substitute for the final AC numerical check.
    """),
    code("""
    def branch_catalog(net):
        branches = []
        for idx, row in net.line.iterrows():
            branches.append((f"line {idx}", int(row.from_bus), int(row.to_bus)))
        for idx, row in net.trafo.iterrows():
            branches.append((f"trafo {idx}", int(row.hv_bus), int(row.lv_bus)))
        return branches

    def flow_incidence_jacobian(net, measured_branches, reference_bus=0):
        nonref = [b for b in net.bus.index if b != reference_bus]
        column = {bus: k for k, bus in enumerate(nonref)}
        rows = []
        for _, from_bus, to_bus in measured_branches:
            row = np.zeros(len(nonref))
            if from_bus != reference_bus:
                row[column[from_bus]] += 1.0
            if to_bus != reference_bus:
                row[column[to_bus]] -= 1.0
            rows.append(row)
        return np.asarray(rows), nonref

    branches = branch_catalog(net)
    branch_graph = nx.Graph()
    for branch in branches:
        branch_graph.add_edge(branch[1], branch[2], record=branch)
    tree_graph = nx.minimum_spanning_tree(branch_graph)
    tree_branches = [branch_graph[u][v]["record"] for u, v in tree_graph.edges]

    placements = {
        "insufficient: one tree branch missing": tree_branches[:-1],
        "measured spanning tree": tree_branches,
        "redundant: all branches": branches,
    }
    rank_rows = []
    for name, selected_branches in placements.items():
        H, unknown_buses = flow_incidence_jacobian(net, selected_branches)
        rank_rows.append({
            "placement": name, "measurements": H.shape[0],
            "unknown angles": H.shape[1], "rank": np.linalg.matrix_rank(H),
            "observable angles": np.linalg.matrix_rank(H) == H.shape[1],
        })
    pd.DataFrame(rank_rows).set_index("placement")
    """, "dc-rank-tests"),
    md("""
    The first placement splits the measured graph into two components, leaving a relative angle shift invisible. A measured spanning tree connects every bus to the reference with 13 independent flow rows. Using all 20 branches adds loop measurements: they are redundant, yet valuable because suitable single-meter losses need not destroy observability.
    """),
    code("""
    H_bad, buses = flow_incidence_jacobian(
        net, placements["insufficient: one tree branch missing"]
    )
    _, singular_values, vh = np.linalg.svd(H_bad, full_matrices=True)
    null_direction = vh[-1]
    pd.Series(null_direction, index=[f"theta_{b}" for b in buses],
              name="unobservable direction")
    """, "unobservable-direction"),
    md("""
    The null direction isolates the angle change that the selected meters cannot see. In larger systems, observability analysis similarly identifies unobservable islands and the pseudo-measurements or real meters needed to connect them.
    """),
    md("""## 4. AC WLS: observe the singular-gain symptom

    Now return to all 27 AC voltage states. Fourteen voltage-magnitude meters alone cannot determine any of the 13 unknown angles. pandapower’s WLS estimator therefore rejects the underobservable measurement set."""),
    code("""
    pp.runpp(net, calculate_voltage_angles=True, numba=False)
    truth = net.res_bus.copy()
    for bus in net.bus.index:
        pp.create_measurement(net, "v", "bus", truth.at[bus, "vm_pu"],
                              0.005, bus, name=f"V{bus}")

    try:
        underobservable = estimate(net, algorithm="wls", init="flat",
                                   tolerance=1e-7, maximum_iterations=10)
    except UserWarning as error:
        underobservable = {"success": False, "reason": str(error)}
    print(f"Only voltage magnitudes ({len(net.measurement)} meters): {underobservable}")
    """, "underobservable-wls"),
    md("""
    A failed estimate is a symptom, not a complete observability study: poor initialization or numerical conditioning can also prevent convergence. A production estimator performs observability analysis before estimation and reports unobservable islands.
    """),
    md("""## 5. Restore observability and redundancy

    Recreate the same IEEE case and the 82-measurement design from Notebook 2. The measurement Jacobian now covers magnitudes and angles across the whole network."""),
    code("""
    net = build_ieee_14_grid()
    pp.runpp(net, calculate_voltage_angles=True, numba=False)
    add_redundant_measurements(net, seed=7)
    redundant_result = estimate(net, algorithm="wls", init="flat",
                                tolerance=1e-7, maximum_iterations=50)
    print(f"Redundant set ({len(net.measurement)} meters): {redundant_result}")
    net.res_bus_est[["vm_pu", "va_degree"]]
    """, "observable-wls"),
    md(r"""
    ## 6. Redundancy, critical measurements, and conditioning

    Define the simple redundancy ratio

    $$\rho=\frac{m}{n_x}.$$

    Here $\rho=82/27$, but this number alone still does not prove observability. Placement controls rank. It is also useful to distinguish:

    - **critical measurement:** removing it makes the system unobservable;
    - **critical set:** removing a particular group makes the system unobservable;
    - **numerical weakness:** $H$ has full rank but $G$ is ill-conditioned, so small meter errors can cause large state errors;
    - **pseudo-measurement:** a forecast or historical estimate used with a relatively large $\sigma$, useful for restoring coverage without pretending it is a precise meter.

    Singular values quantify numerical strength. If $H=U\Sigma V^T$, a very small $\sigma_{\min}(H)$ signals a nearly invisible state direction even when the formal rank is full.
    """),
    code("""
    H_tree, _ = flow_incidence_jacobian(net, placements["measured spanning tree"])
    H_redundant, _ = flow_incidence_jacobian(net, placements["redundant: all branches"])
    conditioning = pd.DataFrame({
        "smallest singular value": [
            np.linalg.svd(H_tree, compute_uv=False).min(),
            np.linalg.svd(H_redundant, compute_uv=False).min(),
        ],
        "condition number": [np.linalg.cond(H_tree), np.linalg.cond(H_redundant)],
    }, index=["measured spanning tree", "all branch-flow meters"])
    conditioning
    """, "conditioning"),
    md("""
    ## 7. Practical observability workflow

    1. Fix the reference angle and determine the state dimension.
    2. Validate network topology and breaker status.
    3. Build the measurement-to-state Jacobian for the available meters.
    4. Test rank and partition any unobservable islands.
    5. Add well-placed real meters or honest, low-weight pseudo-measurements.
    6. Inspect singular values or gain-matrix conditioning, not rank alone.
    7. Only then run WLS and bad-data analysis.

    ### Exercises

    1. Remove a branch from `tree_branches`. Which buses form the unobservable component?
    2. Remove each branch-flow row from `H_redundant` in turn. Which single losses preserve rank 13?
    3. Give one transformer-flow meter ten times the standard deviation of other flow meters. How should its WLS influence change?
    4. Why can a zero-injection bus act like a high-quality constraint, and why must its status be trusted?
    """),
])


OUT.mkdir(parents=True, exist_ok=True)
for filename, nb in [
    ("1.0-power-flow-analysis.ipynb", nb1),
    ("2.0-wls-state-estimation.ipynb", nb2),
    ("3.0-state-estimation-observability.ipynb", nb3),
]:
    nbf.write(nb, OUT / filename)
