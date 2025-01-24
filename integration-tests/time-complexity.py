import argparse
import shutil
import os
import json

from collections import defaultdict
from statistics import mean

from os.path import join
import numpy as np

from scipy.optimize import curve_fit

import pandas as pd  # data manipulation and analysis
from pandas import DataFrame, Series

import plotly.express as px
import plotly.graph_objects as go

boring_colors = ["#EDEDE9", "#D6CCC2", "#F5EBE0", "#E3D5CA", "#d6e2e9"]
real_color = "dodgerblue"
best_color = "limegreen"

def load_df(path: str) -> DataFrame:
    return pd.read_csv(path)


def get_functions() -> dict:
    return {
        r"a": lambda x, a: np.full(x.size, a),
        r"a \cdot \log(n)": lambda x, a: a * np.log(x),
        r"a \cdot n": lambda x, a: a * x,
        r"a \cdot n \log(n)": lambda x, a: a * x * np.log(x),
        r"a \cdot n^2": lambda x, a: a * x ** 2,
        r"a \cdot 2^n": lambda x, a: a * 2 ** x,
        # TODO test more functions, cubic? x^4?
    }

def get_full_name(stack: list[tuple[str, int]]) -> str:
    return "/".join(name for name, _ in stack)

def fold_profiler_data(path: str) -> DataFrame:
    """
    List all json files in data folder and load them
    :param path:
    :return:
    """
    timestats = []

    for f in os.listdir(path):
        if not f.endswith(".json") or not "bestalg" in f or not "bestiter" in f:
            continue

        print("Processing", f)
        with open(join(path, f)) as json_file:
            data = []
            jsondata = json.load(json_file)

        for i in jsondata['timeData']:
            data.append({'when': i['when'], 'enter': i['enter'], 'clazz': i['clazz'], 'method': i['method']})
        data.sort(key=lambda x: x['when'])

        dict_data = defaultdict(list)
        stack = []

        for i in data:
            name = f"{i['clazz']}::{i['method']}"
            if i['enter']:
                stack.append((name, i['when']))
            else:
                current = get_full_name(stack)
                start = stack.pop()
                if name != start[0]:
                    raise Exception(f"Unexpected stack frame: {name} != {start[0]}")
                dict_data[current].append((i['when'] - start[1])/1000000) # Convert nanos to millis

        for k, v in dict_data.items():
            parent, child = k.rsplit("/", 1) if "/" in k else ("", k)
            timestats.append({"instance": jsondata['instanceId'], "component": k, "parent": parent, "child": child, "time": mean(v)})

    return pd.DataFrame(timestats).sort_values(by=['instance', 'component'])


def prepare_df(df: DataFrame, timestats: DataFrame) -> DataFrame:
    exp_instances = timestats['instance'].unique()
    cloned = df[df['id'].isin(exp_instances)]
    cloned = cloned.sort_values(by=['id'], inplace=False)
    cloned = cloned.drop(['id'], axis=1)
    return cloned


def draw_functions_chart(xy, fits, instance_property, component_name):
    fig = px.line()
    fig.add_scatter(x=xy.index, y=xy['time'], name="Real", line=dict(color=real_color))

    for i, fit in enumerate(fits):
        color = boring_colors[i % len(boring_colors)] if i != 0 else best_color
        fig.add_scatter(x=fit.data.x, y=fit.data.y, name=f"${fit.name}$", line=dict(color=color))
        # print(f"Component {c} - Function {k} - {col} - R2: {r2} - {popt} - {dic['fvec']}")

    fig.update_layout(title=rf"$\text{{{component_name} is }}Θ({fits[0].name})$", showlegend=True, xaxis_title=instance_property, yaxis_title="T (ms)")
    fig.show()

class Fit(object):
    name = ""
    instance_prop = ""
    r2 = 0
    popt = {}
    dic = {}
    data = {}

    def __init__(self, name, instance_prop, r2, popt, dic, data):
        self.name = name
        self.instance_prop = instance_prop
        self.r2 = r2
        self.popt = popt
        self.dic = dic
        self.data = data


def calculate_fitting_func(x: Series, y: Series, f_name: str, f, instance_property: str) -> Fit | None:
    popt, pcov, dic, mesg, _ = curve_fit(f, x, y, full_output=True)
    # https://stackoverflow.com/questions/50371428/scipy-curve-fit-raises-optimizewarning-covariance-of-the-parameters-could-not
    # Curve fit puede fallar
    if np.isnan(pcov).any():
        print("NaN values in covariance matrix, skipping")
        return None
    y_estimated = f(x, *popt)
    # residual sum of squares
    ss_res = np.sum((y - y_estimated) ** 2)
    # total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # r-squared
    r2 = 1 - (ss_res / ss_tot)
    data = pd.DataFrame({'x':x, 'y':y_estimated})
    name = f"{f_name}".replace("a", str(round(popt[0], 1)))
    return Fit(name, instance_property, r2, popt, dic, data)

def join_instance_timestats(instances: DataFrame, timestats: DataFrame, component_name: str, instance_property: str):
    x = instances[['id', instance_property]]
    y = timestats[timestats['component'] == component_name][['instance', 'time']]
    xy = x.set_index('id').join(y.set_index('instance'), how='right')
    xy = xy.sort_values(by=[instance_property], inplace=False)
    xy = xy.groupby(instance_property).mean()
    return xy

def calculate_fitting_funcs(instances: DataFrame, timestats: DataFrame, component_name: str, instance_property: str) -> [Fit]:
    fitting_funcs = get_functions()
    xy = join_instance_timestats(instances, timestats, component_name, instance_property)

    fits = []
    x = xy.index.to_numpy()
    y = xy['time'].to_numpy()
    for k, v in fitting_funcs.items():
        fit = calculate_fitting_func(x, y, k, v, instance_property)
        if fit:
            fits.append(fit)

    fits.sort(key=lambda e: e.r2, reverse=True)
    return fits

def find_best_instance_property(instances: DataFrame, timestats: DataFrame, component_name: str) -> [Fit]:
    best_r2 = 0
    best_fits = []
    for instance_property in instances.columns:
        if instance_property == 'id':
            continue


        fits = calculate_fitting_funcs(instances, timestats, component_name, instance_property)
        max_r2 = fits[0].r2
        if max_r2 > best_r2:
            best_fits = fits

    return best_fits


def analyze_complexity(instances: DataFrame, timestats: DataFrame):
    treemap_labels = []
    for component_name in timestats['component'].unique():
        fits = find_best_instance_property(instances, timestats, component_name)
        best = fits[0]
        xy = join_instance_timestats(instances, timestats, component_name, best.instance_prop)
        draw_functions_chart(xy, fits, best.instance_prop, component_name)

        print(f"Component {component_name} performance predicted as Θ({best.name}) by {best.instance_prop} - R2: {best.r2}")
        treemap_labels.append({"component": component_name, "property": best.instance_prop, "function": f"Θ({best.name})", "r2": best.r2})

    treemap_data = timestats.groupby(['component', 'parent', 'child'], as_index=False)['time'].mean()
    treemap_data = treemap_data.merge(pd.DataFrame(treemap_labels), on='component')

    fig = go.Figure()
    fig.add_trace(go.Treemap(
        ids=treemap_data.component,
        labels=treemap_data.child,
        parents=treemap_data.parent,
        customdata=np.stack((treemap_data.time, treemap_data.property, treemap_data.function.str.replace(r'\cdot','⋅'), treemap_data.r2), axis=-1),
        hovertemplate='<b> %{label} </b> <br> Time: %{customdata[0]:.2f} ms <br> Complexity: %{customdata[2]} <br> Where n is: %{customdata[1]} <br> R2: %{customdata[3]:.2f}',
        marker=dict(
            colors=treemap_data.time,
            colorscale='ylorbr',
            colorbar=dict(title='T (ms)'),
            cmid=treemap_data.time.mean(),
            showscale=True
        ),
        maxdepth=3,
        legend="legend"
    ))
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
    )
    fig.show()

def main():
    parser = argparse.ArgumentParser(
        description='Creates a set of instances to use during the experimentation',
        epilog='Created for the Mork project, if useful for your research consider citing the original publication')
    parser.add_argument('-p', '--properties', required=False, default="instance_properties.csv", help="CSV Input file containing instance properties.")
    parser.add_argument('-i', '--data', required=False, default="solutions", help="Path to folder which contains profiler data")
    # parser.add_argument('-o', '--output', required=False, default="output", help="Path to output folder.")

    args = parser.parse_args()

    #shutil.rmtree(args.output, ignore_errors=True)
    #os.mkdir(args.output)
    print(f"Loading CSV {args.properties}")
    instances = load_df(args.properties)
    print("Loading profiler data")
    timestats = fold_profiler_data(args.data)
    #print(f"Preparing CSV data")
    #instances = prepare_df(instances, timestats)
    print(f"Analyzing complexity")
    analyze_complexity(instances, timestats)

    print(f"All done, bye!")


if __name__ == '__main__':
    main()
