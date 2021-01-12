import json
from ioutil import *
import numpy as np
import multiprocessing
import time
import copy
import shutil
from sys import stderr

IDENTITY_CONNECTED_LAYERS = ['ZeroPadding2D', 'BatchNormalization', 'Activation', 'MaxPooling2D',
                             'GlobalAveragePooling2D', 'AveragePooling2D']
MULYIPY_TO_ONE_LAYERS = ['Add']
MAXPOOLING2D = ['MaxPooling2D']
FLATTEN_LAYERS = ["Flatten"]
DENSE_LAYERS = ['Dense']
CONV_LAYERS = ['Conv2D']
INPUT_LAYER = ['InputLayer']

SEQUENTIAL_MODELS = ['lenet1', 'lenet4', 'lenet5', 'vgg19', 'vgg16']
SKIP_CONNECTION_MODELS = ['resnet50']

IMAGENET_MODELS = ['vgg19', 'resnet50']
MNIST_MODELS = ['lenet1', 'lenet4', 'lenet5']

def reverse_dfg(dfg):
    reversed_dfg = {}
    for key_node in dfg:
        for val_node in dfg[key_node]:
            if val_node not in dfg:
                reversed_dfg[val_node] = {key_node}
            else:
                reversed_dfg[val_node].add(key_node)
    for node in reversed_dfg:
        reversed_dfg[node] = list(reversed_dfg[node])
    return reversed_dfg


def aggregate_dfg_node_state(org_graph_state, to_be_aggregated_graph_state_arr):
    if len(org_graph_state) == 0:
        org_graph_state.extend(to_be_aggregated_graph_state_arr[0].copy())
    for graph_state_index in range(0, len(to_be_aggregated_graph_state_arr)):
        graph_state = to_be_aggregated_graph_state_arr[graph_state_index]
        for layer_index in range(0, len(graph_state)):
            if len(graph_state[layer_index]) == 0:
                continue
            for feature_map_index in range(0, len(graph_state[layer_index])):
                if graph_state[layer_index][feature_map_index] == 1:
                    org_graph_state[layer_index][feature_map_index] = 1
    return org_graph_state


# Only the same depth paths can be aggregated
def aggregate_graphs_from_file(org_graph, to_be_aggregated_graph_file_name):
    with open(to_be_aggregated_graph_file_name) as f:
        count = 0
        # len=len(f.read().split("\n"))
        for eachline in f:
            print("INFO, aggregate_graphs_from_file in image: " + str(count + 1))
            dict_of_lists_of_graph = json.loads(eachline)
            aggregate_graphs(org_graph, [dict_of_lists_of_graph])
            count += 1
    return org_graph


# Only full-graphs of the same depth can be aggregated
def aggregate_graphs(org_graph, to_be_aggregated_graph_arr):
    for to_be_aggregated_graph_index in range(0, len(to_be_aggregated_graph_arr)):
        to_be_aggregated_graph = to_be_aggregated_graph_arr[to_be_aggregated_graph_index]
        for key_node in to_be_aggregated_graph:
            if key_node in org_graph:
                following_nodes = copy.deepcopy(to_be_aggregated_graph[key_node])
                org_graph[key_node] += following_nodes
                org_graph[key_node] = list(set(org_graph[key_node]))
            else:
                org_graph[key_node] = copy.deepcopy(to_be_aggregated_graph[key_node])
    return org_graph


def output_dataflow_graphs(covered_dataflow_graphs, file, model='a+'):
    for covered_dataflow_graph in covered_dataflow_graphs:
        write_json_to_file(covered_dataflow_graph, file, 'a+')


# def count_the_number_of_covered_paths(org_graph, start_nodes):
#     # paths = []
#     # with open(file_name, "a+") as fp:
#     cnt = 0
#     for start_node in start_nodes:
#         stack = [[start_node, 0]]
#         while stack:
#             (v, next_child_idx) = stack[-1]
#             if (v not in org_graph) or (next_child_idx >= len(org_graph[v])):
#                 if next_child_idx == 0:
#                     # print(stack)
#                     path = []
#                     for node_idx in range(0, len(stack)):
#                         path.append(stack[node_idx][0])
#
#                     cnt += 1
#                     if cnt % 1000000 == 0:
#                         print("I've seen " + str(cnt) + " paths" + ", the " + str(cnt) + "th path is: " + str(path))
#                     # if file_name:
#                     #     fp.write(str(path)+"\n")
#                     # else:
#                     #     paths.append(path)
#                 stack.pop()
#                 continue
#             next_child = org_graph[v][next_child_idx]
#             stack[-1][1] += 1
#             stack.append([next_child, 0])
#     print("Total paths #: " + str(cnt))
#     return cnt


def add_edge_to_graph(graph, pre_node, connected_node):
    if pre_node not in graph:
        graph[pre_node] = [connected_node]
    elif connected_node not in graph[pre_node]:
        graph[pre_node].append(connected_node)


def get_layer_index_by_node_name(node_name):
    strs = node_name.split("_")
    # return int(strs[1])
    return int(strs[0])


def get_node_idx_by_node_name(node_name):
    strs = node_name.split("_")
    # return int(strs[2])
    return int(strs[1])


def count_independent_reachable_paths(model, data_flow_graph):
    def is_node_in_1thlayer_covered(node, graph):
        for node_index in range(model.layers[1].output_shape[-1]):
            connect_node = get_node_name(1, node_index)
            if connect_node not in graph:
                continue
            for connected_node in graph[connect_node]:
                if node == connected_node:
                    return True
        return False

    paths_num_in_layers = [[0] for i in range(0, len(model.layers))]
    for layer_index in range(0, len(model.layers)):
        layer = model.layers[layer_index]
        output_shape = layer.output_shape
        paths_num_in_layers[layer_index] = [0 for i in range(0, output_shape[-1])]
        for node_index in range(0, output_shape[-1]):
            if layer_index == 0:
                node = get_node_name(layer_index, node_index)
                if is_node_in_1thlayer_covered(node, data_flow_graph):
                    paths_num_in_layers[0][node_index] = 1
            else:
                connect_node = get_node_name(layer_index, node_index)
                if connect_node in data_flow_graph:
                    for connected_node in data_flow_graph[connect_node]:
                        connected_layer_index = get_layer_index_by_node_name(connected_node)
                        connected_nodex_index = get_node_idx_by_node_name(connected_node)
                        paths_num_in_layers[layer_index][node_index] += paths_num_in_layers[connected_layer_index][
                            connected_nodex_index]
    return paths_num_in_layers


def get_node_name(layer_index, featuremap_index=-1):
    if featuremap_index != -1:
        # return "n_" + str(layer_index) + "_" + str(featuremap_index)
        return str(layer_index) + "_" + str(featuremap_index)
    else:
        # return "ly" + str(layer_index)
        return str(layer_index)


def gt_scaled_t(node_vals, t):
    if len(node_vals) == 1:
        if node_vals[0] == 0:
            return [0]
        else:
            return [1]
    min = np.min(node_vals)
    max = np.max(node_vals)
    scaled_val_arr = (node_vals - min) / (max - min)
    ret = [0 for i in range(0, len(node_vals))]
    for i in range(0, len(node_vals)):
        if node_vals[i] != 0:
            if scaled_val_arr[i] > t:
                ret[i] = 1
    return ret


def gte_scaled_t(node_vals, t):
    if len(node_vals) == 1:
        if node_vals[0] == 0:
            return [0]
        else:
            return [1]
    min = np.min(node_vals)
    max = np.max(node_vals)
    scaled_val_arr = (node_vals - min) / (max - min)
    ret = [0 for i in range(0, len(node_vals))]
    for i in range(0, len(node_vals)):
        if node_vals[i] != 0:
            if scaled_val_arr[i] >= t:
                ret[i] = 1
    return ret


def output_all_nodes(node_vals):
    ret = [1 for i in range(0, len(node_vals))]
    return ret


def extract_node_activation_flags_over_one_img(model_json, start_layer_idx, layers_outputs, layers_contributions, t_func, *args):
    from nn_util import get_precursor_layer_idxes, get_featuremap2dense_mapper

    dnn_layers_num = len(model_json["config"]["layers"])
    activation_flags = [[] for i in range(0, dnn_layers_num)]
    for layer_idx in range(dnn_layers_num - 1, start_layer_idx-1, -1):
        node_num = layers_outputs[layer_idx].shape[-1]
        activation_flags[layer_idx] = np.zeros((node_num), dtype=int)

    for layer_idx in range(dnn_layers_num - 1, start_layer_idx, -1):
        class_name = model_json["config"]["layers"][layer_idx]["class_name"]
        pre_layer_idxes = get_precursor_layer_idxes(model_json, layer_idx)
        node_num = layers_outputs[layer_idx].shape[-1]
        if class_name in IDENTITY_CONNECTED_LAYERS:
            pre_layer_idx = pre_layer_idxes[0]
            activation_flags[pre_layer_idx] = activation_flags[layer_idx]
        elif class_name in MULYIPY_TO_ONE_LAYERS:
            for pre_layer_idx in pre_layer_idxes:
                activation_flags[pre_layer_idx] = activation_flags[layer_idx]
                pre_node_num = layers_outputs[pre_layer_idx].shape[-1]
                for pre_node_idx in range(0, pre_node_num):
                    if np.all(layers_outputs[pre_layer_idx][pre_node_idx] == 0):
                        activation_flags[pre_layer_idx][pre_node_idx] = 0

        elif class_name in CONV_LAYERS:
            pre_layer_idx = pre_layer_idxes[0]
            for node_idx in range(0, node_num):
                contributions_to_node = layers_contributions[layer_idx][..., node_idx]
                if np.all(contributions_to_node==0):
                    continue
                contri_num = contributions_to_node.shape[-1]
                contri_means = []
                for contri_idx in range(0, contri_num):
                    contri_means.append(np.mean(contributions_to_node[..., contri_idx]))
                contri_means = np.array(contri_means)
                contri_flags = t_func(contri_means, args[0])
                activation_flags[pre_layer_idx] = np.logical_or(activation_flags[pre_layer_idx],
                                                                np.array(contri_flags))
                if np.all(activation_flags[pre_layer_idx] == True):
                    break

        elif class_name in DENSE_LAYERS:
            pre_layer_idx = pre_layer_idxes[0]
            if layer_idx == dnn_layers_num-1:
                node_idx = np.argmax(layers_outputs[layer_idx])
                activation_flags[layer_idx][node_idx]=1

                contributions_to_node = layers_contributions[layer_idx][..., node_idx]
                contri_flags = t_func(contributions_to_node, args[0])
                activation_flags[pre_layer_idx] = np.array(contri_flags)
            else:
                for node_idx in range(0, node_num):
                    contributions_to_node = layers_contributions[layer_idx][..., node_idx]
                    if np.all(contributions_to_node == 0):
                        continue
                    contri_flags = t_func(contributions_to_node, args[0])
                    activation_flags[pre_layer_idx] = np.logical_or(activation_flags[pre_layer_idx],
                                                                    np.array(contri_flags))
                    if np.all(activation_flags[pre_layer_idx] == True):
                        break
        elif class_name in FLATTEN_LAYERS:
            pre_layer_idx = pre_layer_idxes[0]
            pre_node_num = layers_outputs[pre_layer_idx].shape[-1]
            for pre_node_idx in range(0, pre_node_num):
                pre_layer_feature_maps = layers_outputs[pre_layer_idx]
                flatten_map_arr = get_featuremap2dense_mapper(pre_layer_feature_maps)
                if np.all(pre_layer_feature_maps[..., pre_node_idx] == 0):
                    continue
                for w_idx in range(0, flatten_map_arr.shape[1]):
                    for h_idx in range(0, flatten_map_arr.shape[2]):
                        cur_node_idx = int(flatten_map_arr[pre_node_idx][w_idx][h_idx])
                        if activation_flags[layer_idx][cur_node_idx]:
                            activation_flags[pre_layer_idx][pre_node_idx] = activation_flags[layer_idx][cur_node_idx]

    return activation_flags


def extrac_one_covered_dfg(model_json, all_layers_outs_of_one_img, covered_dfg, pred, exclude_layers, t_func, *args):
    from nn_util import get_precursor_layer_idxes, get_featuremap2dense_mapper

    layers = model_json["config"]["layers"]
    for layer_idx in range(len(layers) - 1, -1, -1):
        if layer_idx in exclude_layers:
            continue
        class_name = model_json["config"]["layers"][layer_idx]["class_name"]
        precursor_layer_idxes = get_precursor_layer_idxes(model_json, layer_idx)
        output_node_num = all_layers_outs_of_one_img[layer_idx].shape[-1]
        if class_name in IDENTITY_CONNECTED_LAYERS or class_name in MULYIPY_TO_ONE_LAYERS:
            for node_idx in range(0, output_node_num):
                cur_node = get_node_name(layer_idx, node_idx)
                if cur_node not in covered_dfg:
                    continue
                if np.mean(all_layers_outs_of_one_img[layer_idx][..., node_idx]) == 0:
                    print("ERROR, cur node {0}_{1} should not be zero!".format(layer_idx, node_idx))
                    continue
                for precursor_layer_idx in precursor_layer_idxes:
                    if np.mean(all_layers_outs_of_one_img[precursor_layer_idx][..., node_idx]) == 0:
                        continue
                    pre_node = get_node_name(precursor_layer_idx, node_idx)
                    add_edge_to_graph(covered_dfg, pre_node, cur_node)
        elif class_name in DENSE_LAYERS + CONV_LAYERS:
            precursor_layer_idx = precursor_layer_idxes[0]
            for node_idx in range(0, output_node_num):
                cur_node = get_node_name(layer_idx, node_idx)
                if cur_node not in covered_dfg:
                    continue
                if len(pred[layer_idx]['c_i'][node_idx]) == 0:
                    continue
                c_ins = pred[layer_idx]['c_i'][node_idx]
                if args:
                    flags = t_func(c_ins, args[0])
                else:
                    flags = t_func(c_ins)
                for pre_node_idx in range(0, len(pred[layer_idx]['c_i'][node_idx])):
                    if flags[pre_node_idx] == 1:
                        pre_node_val = all_layers_outs_of_one_img[precursor_layer_idx][..., pre_node_idx]
                        if np.all(pre_node_val == 0):
                            continue
                        pre_node = get_node_name(precursor_layer_idx, pre_node_idx)
                        add_edge_to_graph(covered_dfg, pre_node, cur_node)
        elif class_name in FLATTEN_LAYERS:
            precursor_layer_idx = precursor_layer_idxes[0]
            pre_layer_n_num = all_layers_outs_of_one_img[precursor_layer_idx].shape[-1]
            for pre_layer_n_idx in range(0, pre_layer_n_num):
                pre_layer_feature_maps = all_layers_outs_of_one_img[precursor_layer_idx]
                flatten_map_arr = get_featuremap2dense_mapper(pre_layer_feature_maps)
                if np.all(pre_layer_feature_maps[..., pre_layer_n_idx] == 0):
                    continue
                pre_node = get_node_name(precursor_layer_idxes[0], pre_layer_n_idx)
                for w_idx in range(0, flatten_map_arr.shape[1]):
                    for h_idx in range(0, flatten_map_arr.shape[2]):
                        cur_layer_n_idx = int(flatten_map_arr[pre_layer_n_idx][w_idx][h_idx])
                        cur_node = get_node_name(layer_idx, cur_layer_n_idx)
                        if cur_node not in covered_dfg:
                            continue
                        if all_layers_outs_of_one_img[layer_idx][..., cur_layer_n_idx] == 0:
                            print("ERROR, the cur {0}-th layer {1}-th neuron is zero and cur class name is {2}.".format(
                                layer_idx, node_idx, class_name))
                            continue
                        add_edge_to_graph(covered_dfg, pre_node, cur_node)
    return covered_dfg


def extract_covered_dfgs_from_contris_file(model_json, contris_file, all_layers_outs=[], t_func=gte_scaled_t, *args):
    contris = read_pkls_from_file(contris_file)
    return extract_covered_dfgs(model_json, contris, all_layers_outs, t_func, args)

def extract_covered_dfgs(model_json, contris, all_layers_outs=[], t_func=gte_scaled_t, *args):
    covered_dataflow_graphs = [{} for i in range(0, len(contris))]
    pred_labels = []
    for img_idx in range(0, len(contris)):
        argmax_idx = np.argmax(all_layers_outs[-1][img_idx])
        pred_labels.append(argmax_idx)
        pre_node = get_node_name(len(contris[img_idx]) - 1, argmax_idx)
        add_edge_to_graph(covered_dataflow_graphs[img_idx], pre_node, '')

    # opt: 1.1 Extracting dfgs in parallel
    pool = multiprocessing.Pool()
    rets = []
    model_len = len(model_json["config"]["layers"])
    for img_idx in range(0, len(contris)):
        all_layers_outs_of_one_img = []
        for layer_idx in range(0, model_len):
            all_layers_outs_of_one_img.append(all_layers_outs[layer_idx][img_idx])
        covered_dfg = covered_dataflow_graphs[img_idx]
        pred = contris[img_idx]
        ret = pool.apply_async(extrac_one_covered_dfg,
                               (model_json, all_layers_outs_of_one_img, covered_dfg, pred, [], t_func, args))
        rets.append(ret)
    pool.close()
    pool.join()
    for img_idx in range(0, len(contris)):
        covered_dataflow_graphs[img_idx] = rets[img_idx].get()

    for img_idx in range(0, len(contris)):
        argmax_idx = np.argmax(all_layers_outs[-1][img_idx])
        pre_node = get_node_name(len(contris[img_idx]) - 1, argmax_idx)
        covered_dataflow_graphs[img_idx].pop(pre_node)
    return pred_labels, covered_dataflow_graphs


def get_start_and_end_img_idx_by_filename(filename):
    filename = os.path.split(filename)[-1]
    start_idx, end_idx = filename.split('_')[-1].split('.')[0].split('-')
    return int(start_idx), int(end_idx)


# def get_pre_connected_include_layer(model_json, layer_idx):
#     include_layers = DENSE_LAYERS + CONV_LAYERS + INPUT_LAYER + FLATTEN_LAYERS
#     from OutputDNNPreds import get_precursor_layer_idxes
#     def is_include_layer(layer_idx):
#         class_name = model_json["config"]["layers"][layer_idx]["class_name"]
#         if class_name in include_layers:
#             return True
#         return False
#
#     pre_layer_idxes = get_precursor_layer_idxes(model_json, layer_idx)
#     layer_queue = pre_layer_idxes
#     pre_include_layers = []
#     while layer_queue:
#         poped_layer = layer_queue.pop(0)
#         if is_include_layer(poped_layer):
#             pre_include_layers.append(poped_layer)
#         else:
#             pre_pre_layer_idxes = get_precursor_layer_idxes(model_json, poped_layer)
#             layer_queue += pre_pre_layer_idxes
#     return pre_include_layers


def get_sliding_layers(model_json, model_name):
    sliding_layers = []
    if model_name in SEQUENTIAL_MODELS:
        if model_name == 'resnet50':
            for layer_idx in range(1, len(model_json["config"]["layers"])):
                class_name = model_json["config"]["layers"][layer_idx]["class_name"]
                if class_name in MULYIPY_TO_ONE_LAYERS:
                    sliding_layers.append([layer_idx])
    elif model_name in SEQUENTIAL_MODELS:
        if model_name == 'vgg19':
            for layer_idx in range(1, len(model_json["config"]["layers"])):
                class_name = model_json["config"]["layers"][layer_idx]["class_name"]
                if class_name in DENSE_LAYERS + MAXPOOLING2D:
                    sliding_layers.append([layer_idx])
        elif model_name in ['lenet1', 'lenet4', 'lenet5']:
            for layer_idx in range(1, len(model_json["config"]["layers"])):
                class_name = model_json["config"]["layers"][layer_idx]["class_name"]
                if class_name in DENSE_LAYERS + CONV_LAYERS:
                    sliding_layers.append([layer_idx])
    return sliding_layers


def get_included_layers(model_json, model_name):
    from nn_util import get_precursor_layer_idxes
    def is_included_layer(layer, include_layers):
        for layers in include_layers:
            if layer in layers:
                return True
        return False

    def take_first(elem):
        return elem[0]

    include_layers = []
    include_layers.append([0])
    if model_name in SEQUENTIAL_MODELS:
        for layer_idx in range(1, len(model_json["config"]["layers"])):
            class_name = model_json["config"]["layers"][layer_idx]["class_name"]
            if class_name in DENSE_LAYERS + CONV_LAYERS:
                include_layers.append([layer_idx])
    elif model_name in SKIP_CONNECTION_MODELS:
        if model_name == 'resnet50':
            for layer_idx in range(1, len(model_json["config"]["layers"])):
                class_name = model_json["config"]["layers"][layer_idx]["class_name"]
                if class_name in MULYIPY_TO_ONE_LAYERS:
                    pre_pre_layer_idxes = get_precursor_layer_idxes(model_json, layer_idx)
                    tmp_include_layers = []
                    for pre_layer_idx in pre_pre_layer_idxes:
                        pre_class_name = model_json["config"]["layers"][pre_layer_idx]["class_name"]
                        if pre_class_name == 'BatchNormalization':
                            pre_pre_layer_idx = get_precursor_layer_idxes(model_json, pre_layer_idx)[0]
                            pre_pre_class_name = model_json["config"]["layers"][pre_pre_layer_idx]["class_name"]
                            if pre_pre_class_name in CONV_LAYERS:
                                tmp_include_layers.append(pre_pre_layer_idx)
                    include_layers.append(tmp_include_layers)
            for layer_idx in range(1, len(model_json["config"]["layers"])):
                class_name = model_json["config"]["layers"][layer_idx]["class_name"]
                if class_name in DENSE_LAYERS + CONV_LAYERS + MULYIPY_TO_ONE_LAYERS:
                    if is_included_layer(layer_idx, include_layers):
                        continue
                    else:
                        include_layers.append([layer_idx])
            include_layers = sorted(include_layers, key=take_first)
    else:
        print("ERROR, the model {0} is not support!".format(model_name))
    return include_layers


def count_each_layer_total_edges_num(model_json, input_shapes, output_shapes):
    each_layer_total_edges_num = {}
    for layer_idx in range(0, len(input_shapes)):
        each_layer_total_edges_num[layer_idx] = 0
        class_name = model_json["config"]["layers"][layer_idx]["class_name"]
        input_shape = input_shapes[layer_idx]
        output_shape = output_shapes[layer_idx]
        if class_name in DENSE_LAYERS + CONV_LAYERS:
            each_layer_total_edges_num[layer_idx] = input_shape[-1] * output_shape[-1]
        elif class_name in IDENTITY_CONNECTED_LAYERS:
            each_layer_total_edges_num[layer_idx] = input_shape[-1]
        elif class_name in MULYIPY_TO_ONE_LAYERS:
            for input_shape_of_one_pre_layer in input_shape:
                each_layer_total_edges_num[layer_idx] += input_shape_of_one_pre_layer[-1]
        elif class_name in FLATTEN_LAYERS:
            each_layer_total_edges_num[layer_idx] = output_shape[-1]
    return each_layer_total_edges_num


def calculate_each_layer_covered_node_and_edge(model, aggregated_dfg, exclude_layers=[], node_related_exclude_layers=[], all_pre_related_layers=[]):
    from nn_util import get_precursor_layer_idxes

    model_json = json.loads(model.to_json())

    each_layer_covered_nodes_num = {}
    layer_count = len(model.layers)
    for layer_idx in range(0, layer_count - 1):
        if layer_idx in node_related_exclude_layers:
            continue
        each_layer_covered_nodes_num[layer_idx] = 0
        for node_idx in range(0, model.layers[layer_idx].output_shape[-1]):
            node_name = get_node_name(layer_idx, node_idx)
            if node_name in aggregated_dfg:
                each_layer_covered_nodes_num[layer_idx] += 1
    output_layer_covered_nodes = set()
    for node_idx in range(0, model.layers[layer_count - 2].output_shape[-1]):
        node_name = get_node_name(layer_count - 2, node_idx)
        if node_name in aggregated_dfg:
            output_layer_covered_nodes |= set(aggregated_dfg[node_name])
    each_layer_covered_nodes_num[layer_count - 1] = len(output_layer_covered_nodes)

    each_layer_covered_edges_num = {}
    for layer_idx in range(1, layer_count):
        if layer_idx in exclude_layers:
            continue
        each_layer_covered_edges_num[layer_idx] = 0
        pre_layer_idxes = get_precursor_layer_idxes(model_json, layer_idx)
        for pre_layer_idx in pre_layer_idxes:
            for in_node_idx in range(0, model.layers[pre_layer_idx].output_shape[-1]):
                pre_node = get_node_name(pre_layer_idx, in_node_idx)
                if pre_node in aggregated_dfg:
                    if pre_layer_idx in all_pre_related_layers:
                        for following_node in aggregated_dfg[pre_node]:
                            following_layer = get_layer_index_by_node_name(following_node)
                            if following_layer == layer_idx:
                                each_layer_covered_edges_num[layer_idx] += 1
                    else:
                        each_layer_covered_edges_num[layer_idx] += len(aggregated_dfg[pre_node])

    return each_layer_covered_nodes_num, each_layer_covered_edges_num


def calculate_covered_ndc_and_ec(model, each_layer_covered_nodes_num, each_layer_covered_edges_num,
                                 each_layer_edges_num, exclude_layers=[], node_related_exclude_layers=[]):
    if len(each_layer_edges_num) == 0:
        each_layer_edges_num = count_each_layer_total_edges_num(model)

    ndc_i = {}
    covered_nodes_num = 0
    all_nodes_num = 0
    for layer_idx in range(0, len(model.layers)):
        if layer_idx in node_related_exclude_layers:
            continue
        layer_nodes_num = model.layers[layer_idx].output_shape[-1]
        covered_nodes_num += each_layer_covered_nodes_num[layer_idx]
        all_nodes_num += layer_nodes_num
        ndc_i[layer_idx] = each_layer_covered_nodes_num[layer_idx] / layer_nodes_num
    ndc = covered_nodes_num / all_nodes_num

    ec_i = {}
    all_edges_num = 0
    covered_edges_num = 0
    for layer_idx in range(1, len(model.layers)):
        if layer_idx in exclude_layers:
            continue
        covered_edges_num += each_layer_covered_edges_num[layer_idx]
        all_edges_num += each_layer_edges_num[layer_idx]
        ec_i[layer_idx] = each_layer_covered_edges_num[layer_idx] / each_layer_edges_num[layer_idx]
    ec = covered_edges_num / all_edges_num

    return ndc, ndc_i, ec, ec_i


def calculate_covered_ndc_and_ec_of_one_category(model, each_layer_covered_nodes_num, each_layer_covered_edges_num,
                                                 each_layer_edges_num, exclude_layers=[],
                                                 node_related_exclude_layers=[]):
    if len(each_layer_edges_num) == 0:
        each_layer_edges_num = count_each_layer_total_edges_num(model)

    ndc_i = {}

    covered_nodes_num = 0

    all_nodes_num = 0
    layers_num = len(model.layers)
    for layer_idx in range(0, layers_num):
        if layer_idx in node_related_exclude_layers:
            continue
        if layer_idx == layers_num - 1:
            covered_nodes_num += 1
            layer_nodes_num = 1
            all_nodes_num += 1
        else:
            covered_nodes_num += each_layer_covered_nodes_num[layer_idx]
            layer_nodes_num = model.layers[layer_idx].output_shape[-1]
            all_nodes_num += layer_nodes_num
        ndc_i[layer_idx] = each_layer_covered_nodes_num[layer_idx] / layer_nodes_num
    ndc = covered_nodes_num / all_nodes_num

    ec_i = {}
    covered_edges_num = 0
    all_edges_num = 0
    for layer_idx in range(1, layers_num):
        if layer_idx in exclude_layers:
            continue
        covered_edges_num += each_layer_covered_edges_num[layer_idx]
        all_edges_num += each_layer_edges_num[layer_idx]
        ec_i[layer_idx] = each_layer_covered_edges_num[layer_idx] / each_layer_edges_num[layer_idx]
    ec = covered_edges_num / all_edges_num

    return ndc, ndc_i, ec, ec_i


# def calculate_each_layer_edge_coverage(model, aggregated_dfg, each_layer_edges_num=[], exclude_layers=[]):
#     if len(each_layer_edges_num) == 0:
#         each_layer_edges_num = count_each_layer_total_edges_num(model)
#
#     from OutputDNNPreds import get_precursor_layer_idxes
#
#     model_json = json.loads(model.to_json())
#     each_layer_covered_edges_num = {}
#     ec_i = {}
#     for layer_idx in range(1, len(model.layers)):
#         if layer_idx in exclude_layers:
#             continue
#         each_layer_covered_edges_num[layer_idx] = 0
#         pre_layer_idxes = get_precursor_layer_idxes(model_json, layer_idx)
#         for pre_layer_idx in pre_layer_idxes:
#             for in_node_idx in range(0, model.layers[pre_layer_idx].output_shape[-1]):
#                 pre_node = get_node_name(pre_layer_idx, in_node_idx)
#                 if pre_node in aggregated_dfg:
#                     each_layer_covered_edges_num[layer_idx]+=len(aggregated_dfg[pre_node])
#         ec_i[layer_idx] = each_layer_covered_edges_num[layer_idx] / each_layer_edges_num[layer_idx]
#     return ec_i


# def get_k_reachable_paths_of_k_subdfg(org_dfg, include_layers, output_shapes):
#     # import time
#     start_layer_idx = include_layers[0]
#     end_layer_idx = include_layers[-1]
#     all_k_reachable_path = {}
#     # cnt = 0
#     for out_node_idx in range(0, output_shapes[start_layer_idx][-1]):
#         start_node = get_node_name(start_layer_idx, out_node_idx)
#
#         if start_node not in org_dfg:
#             continue
#         # print("INFO, get krps of k-win from node: {0}.".format(start_node))
#         # st=time.time()
#         all_k_reachable_path[start_node] = set()
#         stack = [[start_node, 0]]
#         while stack:
#             (v, next_child_idx) = stack[-1]
#             layer_idx_of_v = get_layer_index_by_node_name(v)
#             if (layer_idx_of_v == end_layer_idx) or (next_child_idx >= len(org_dfg[v])):
#                 if next_child_idx == 0:
#                     krp = []
#                     for node_idx in range(1, len(stack)):
#                         # if get_layer_index_by_node_name(stack[node_idx][0]) in include_layers:
#                         krp.append(stack[node_idx][0])
#                     krp = '-'.join(krp)
#                     all_k_reachable_path[start_node].add(krp)
#                 stack.pop()
#                 continue
#             # since the jump connection, the next connected node may out of the sliding windows
#             if get_layer_index_by_node_name(org_dfg[v][next_child_idx]) > end_layer_idx:
#                 # print(v, '->', org_dfg[v])
#                 # print(org_dfg[v][next_child_idx], '>', end_layer_idx)
#                 stack[-1][1] += 1
#                 continue
#             next_child = org_dfg[v][next_child_idx]
#             stack[-1][1] += 1
#             stack.append([next_child, 0])
#         # et = time.time()
#         # print("dur.", et-st)
#     return all_k_reachable_path


def get_k_reachable_paths_of_k_subdfg(sub_graph, start_layer_idx, output_shapes):
    # import time
    all_k_reachable_path = {}
    # cnt = 0
    for out_node_idx in range(0, output_shapes[start_layer_idx][-1]):
        start_node = get_node_name(start_layer_idx, out_node_idx)
        if start_node not in sub_graph:
            continue
        print("INFO, get krps of k-win from node: {0}.".format(start_node))
        # st = time.time()
        all_k_reachable_path[start_node] = set()
        stack = [[start_node, 0]]
        while stack:
            (v, next_child_idx) = stack[-1]
            if (v not in sub_graph) or (next_child_idx >= len(sub_graph[v])):
                if next_child_idx == 0:
                    krp = []
                    for node_idx in range(1, len(stack)):
                        # if get_layer_index_by_node_name(stack[node_idx][0]) in include_layers:
                        krp.append(stack[node_idx][0])
                    krp = '-'.join(krp)
                    all_k_reachable_path[start_node].add(krp)
                stack.pop()
                continue
            next_child = sub_graph[v][next_child_idx]
            stack[-1][1] += 1
            stack.append([next_child, 0])
        # et = time.time()
        # print("dur.", et - st)
    return all_k_reachable_path


# include start layer and end layer
def get_sub_graph(org_graph, start_layer_idxes, end_layer_idxes, output_shapes):
    # sub_graph = {}
    # for layer_idx in range(start_layer_idx, end_layer_idx + 1):
    #     for n_idx in range(0, output_shapes[layer_idx][-1]):
    #         node = get_node_name(layer_idx, n_idx)
    #         if node in org_graph:
    #             sub_graph[node] = org_graph[node]
    # return sub_graph
    sub_graph = {}
    start_layer_idx = min(start_layer_idxes)
    end_layer_idx = max(end_layer_idxes)
    for layer_idx in range(start_layer_idx, end_layer_idx):
        for n_idx in range(0, output_shapes[layer_idx][-1]):
            node = get_node_name(layer_idx, n_idx)
            if node in org_graph:
                sub_graph[node] = org_graph[node]
    return sub_graph


# def is_graph_is_fully_covered(model_json, covered_dfg, input_shapes,output_shapes, include_layers, sliding_win_width, stride):
#     for idx_idx in range(len(include_layers) - 1, -1, -stride):
#         if idx_idx + 1 - sliding_win_width < 0:
#             break
#         start_layer_idx = include_layers[max(idx_idx + 1 - sliding_win_width, 0)]
#         end_layer_idx = include_layers[idx_idx]
#         for layer_idx in range(start_layer_idx+1, end_layer_idx + 1):
#             class_name = model_json["config"]["layers"][layer_idx]["class_name"]
#             if class_name in DENSE_LAYERS+CONV_LAYERS:
#                 for n_idx in range(0, output_shapes[layer_idx][-1]):
#                     node = get_node_name(layer_idx, n_idx)
#                     if node not in covered_dfg:
#                         return False
#                     if len(covered_dfg[node])!=input_shapes[layer_idx][-1]
#     return True


# we only need to compute the include layer in the sliding window.
def get_k_reachable_paths_of_dnn(model_json, model_name, covered_dfgs, output_shapes, sliding_win_width, stride):
    include_layers = get_included_layers(model_json, model_name)
    sliding_wins = [{} for i in range(0, len(covered_dfgs))]

    # if the sliding window is fully covered, we will not extract its krps to save compute-time.
    for covered_dfg_idx in range(0, len(covered_dfgs)):
        covered_dfg = covered_dfgs[covered_dfg_idx]
        for idx_idx in range(len(include_layers) - 1, -1, -stride):
            if idx_idx + 1 - sliding_win_width < 0:
                break
            start_layer_idxes = include_layers[max(idx_idx + 1 - sliding_win_width, 0)]
            end_layer_idxes = include_layers[idx_idx]
            print("Extracting krps in sub-dfg G_{0-1} of {2}-th dfg.".format(start_layer_idxes, end_layer_idxes,
                                                                             covered_dfg_idx))
            sub_graph = get_sub_graph(covered_dfg, start_layer_idxes, end_layer_idxes, output_shapes)
            sliding_win_key = str(start_layer_idxes + '-' + end_layer_idxes)
            start_layer_idx = min(start_layer_idxes)
            sliding_wins[covered_dfg_idx][sliding_win_key] = get_k_reachable_paths_of_k_subdfg(sub_graph,
                                                                                               start_layer_idx,
                                                                                               output_shapes)
    return sliding_wins


def aggregate_krps_of_dnns(aggregated_krps, krps_of_dfgs):
    for krps_of_dfg in krps_of_dfgs:
        for krp_key in krps_of_dfg:
            if krp_key not in aggregated_krps:
                aggregated_krps[krp_key] = krps_of_dfg[krp_key]
            else:
                for start_node in krps_of_dfg[krp_key]:
                    if start_node not in aggregated_krps[krp_key]:
                        aggregated_krps[krp_key][start_node] = krps_of_dfg[krp_key][start_node]
                    else:
                        aggregated_krps[krp_key][start_node] |= krps_of_dfg[krp_key][start_node]
    return aggregated_krps


# def count_krps_num_of_singe_k_sliding_win(model_json, input_shapes, output_shapes, start_layer_idx, end_layer_idx):
#     from OutputDNNPreds import get_precursor_layer_idxes
#     path_num = [[] for i in range(0, end_layer_idx - start_layer_idx + 1)]
#     # path_num[0] = np.ones(output_shapes[start_layer_idx][-1])
#     path_num[0] = [1 for i in range(0, output_shapes[start_layer_idx][-1])]
#     for layer_idx in range(start_layer_idx + 1, end_layer_idx + 1):
#         # path_num[layer_idx] = np.zeros(output_shapes[layer_idx][-1])
#         class_name = model_json["config"]["layers"][layer_idx]["class_name"]
#         path_num[layer_idx - start_layer_idx] = [0 for i in range(0, output_shapes[layer_idx][-1])]
#         pre_layer_idxes = get_precursor_layer_idxes(model_json, layer_idx)
#         pre_layer_idx_idx = 0
#         for pre_layer_idx in pre_layer_idxes:
#             if pre_layer_idx < start_layer_idx:
#                 continue
#             if class_name in IDENTITY_CONNECTED_LAYERS + FLATTEN_LAYERS:
#                 for out_node_idx in range(0, output_shapes[layer_idx][-1]):
#                     path_num[layer_idx - start_layer_idx][out_node_idx] = path_num[pre_layer_idx - start_layer_idx][0]
#             elif class_name in DENSE_LAYERS + CONV_LAYERS:
#                 for out_node_idx in range(0, output_shapes[layer_idx][-1]):
#                     for in_node_idx in range(0, input_shapes[layer_idx][-1]):
#                         path_num[layer_idx - start_layer_idx][out_node_idx] += \
#                             path_num[pre_layer_idx - start_layer_idx][in_node_idx]
#             elif class_name in MULYIPY_TO_ONE_LAYERS:
#                 for in_node_idx in range(0, input_shapes[layer_idx][pre_layer_idx_idx][-1]):
#                     path_num[layer_idx - start_layer_idx][in_node_idx] = path_num[pre_layer_idx - start_layer_idx][
#                         in_node_idx]
#             pre_layer_idx_idx += 1
#     total_krp_num = 0
#     for out_node_idx in range(0, output_shapes[end_layer_idx][-1]):
#         total_krp_num += path_num[end_layer_idx - start_layer_idx][out_node_idx]
#     return total_krp_num


def count_krps_num_of_a_k_sliding_win(model_json, output_shapes, start_layer_idxes, end_layer_idxes):
    from nn_util import get_precursor_layer_idxes
    k_rps_num_arr = {}
    for start_layer_idx in start_layer_idxes:
        k_rps_num_arr[start_layer_idx] = output_shapes[start_layer_idx][-1]
    start_layer_idx = min(start_layer_idxes)
    end_layer_idx = max(end_layer_idxes)
    for layer_idx in range(start_layer_idx + 1, end_layer_idx + 1):
        class_name = model_json["config"]["layers"][layer_idx]["class_name"]
        pre_layer_idxes = get_precursor_layer_idxes(model_json, layer_idx)
        if class_name in IDENTITY_CONNECTED_LAYERS:
            pre_layer_idx = pre_layer_idxes[0]
            if pre_layer_idx < start_layer_idx:
                continue
            k_rps_num_arr[layer_idx] = k_rps_num_arr[pre_layer_idx]
        elif class_name in MULYIPY_TO_ONE_LAYERS:
            k_rps_num_arr[layer_idx] = 0
            for pre_layer_idx in pre_layer_idxes:
                if pre_layer_idx < start_layer_idx:
                    continue
                k_rps_num_arr[layer_idx] += k_rps_num_arr[pre_layer_idx]
        elif class_name in CONV_LAYERS + DENSE_LAYERS:
            pre_layer_idx = pre_layer_idxes[0]
            if pre_layer_idx < start_layer_idx:
                k_rps_num_arr[layer_idx] = output_shapes[layer_idx][-1]
                continue
            pre_class_name = model_json["config"]["layers"][pre_layer_idx]["class_name"]
            if pre_class_name in FLATTEN_LAYERS:
                pre_pre_layer_idx = get_precursor_layer_idxes(model_json, pre_layer_idx)[0]
                # if pre_pre_layer_idx < start_layer_idx:
                #     continue
                k_rps_num_arr[layer_idx] = k_rps_num_arr[pre_pre_layer_idx] * output_shapes[layer_idx][-1]
            else:
                k_rps_num_arr[layer_idx] = k_rps_num_arr[pre_layer_idxes[0]] * output_shapes[layer_idx][-1]

    if len(end_layer_idxes) > 1:
        temp_sum = 0
        for temp_end_layer_idx in end_layer_idxes:
            temp_sum += k_rps_num_arr[temp_end_layer_idx]
            pre_layer_idx = get_precursor_layer_idxes(model_json, temp_end_layer_idx)[0]
            if pre_layer_idx < start_layer_idx:
                temp_max = k_rps_num_arr[max(k_rps_num_arr, key=k_rps_num_arr.get)]
                return temp_max
        return temp_sum
    return k_rps_num_arr[end_layer_idx]


def count_krps_num_of_each_k_sliding_win(model_json, model_name, output_shapes, sliding_win_width, stride):
    include_layers = get_included_layers(model_json, model_name)
    krps_num = {}
    for idx in range(len(include_layers) - 1, -1, -stride):
        if idx + 1 - sliding_win_width < 0:
            break
        st_idx = max(idx + 1 - sliding_win_width, 0)
        end_idx = idx
        start_layer_idxes = include_layers[st_idx]
        end_layer_idxes = include_layers[end_idx]
        sliding_win_key = str(include_layers[st_idx]) + '-' + str(include_layers[end_idx])
        krps_num[sliding_win_key] = count_krps_num_of_a_k_sliding_win(model_json, output_shapes, start_layer_idxes,
                                                                      end_layer_idxes)
    return krps_num


def get_each_layer_valid_node_num_of_a_dfg(output_shapes, covered_dfg):
    node_cnt = [0 for i in range(0, len(output_shapes))]
    for layer_idx in range(0, len(output_shapes)):
        for out_n_idx in range(0, output_shapes[layer_idx][-1]):
            node = get_node_name(layer_idx, out_n_idx)
            if node in covered_dfg:
                node_cnt[layer_idx] += 1
    return node_cnt


def get_category_by_covered_dfg(covered_dfg, output_shapes):
    output_layer_idx = len(output_shapes) - 1
    pre_layer_idx = output_layer_idx - 1
    for pre_node_idx in range(0, output_shapes[pre_layer_idx][-1]):
        pre_node = get_node_name(pre_layer_idx, pre_node_idx)
        if pre_node in covered_dfg:
            nodes = covered_dfg[pre_node]
            if len(covered_dfg[pre_node]) != 1:
                print("ERROR, the number of final node is more than one!")
            category = get_node_idx_by_node_name(nodes[0])
            return category
    return -1


# def dfg_main():
#     import data_util
#     import nn_contribution_util
#     from keras.models import load_model
#     from keras.models import Model
#     from keras.applications.vgg19 import VGG19
#     from keras.applications.resnet50 import ResNet50
#
#     # # lenet1
#     # model_path = './models/lenet1-relu'
#     # model_name = 'lenet1'
#     # # lenet4
#     # model_path = './models/lenet4-relu'
#     # model_name = 'lenet4'
#     # lenet5
#     model_path = './models/lenet5-relu'
#     model_name = 'lenet5'
#
#     model = load_model(model_path)
#     data_type = 'mnist'
#     category_count = 10
#
#     # model = ResNet50(weights='imagenet')
#     # model_name = 'resnet50'
#     # model = VGG19(weights='imagenet')
#     # model_name = 'vgg19'
#     # data_type = 'imagenet'
#     # category_count = 1000
#
#     model_json = json.loads(model.to_json())
#     import nn_util
#     input_shapes = nn_util.get_dnn_each_layer_input_shape(model_json)
#     output_shapes = nn_util.get_dnn_each_layer_output_shape(model)
#     base_dir = "./outputs/" + data_type + "/" + model_name + "/contris/"
#     threshold = 0
#     t_path = "gt_" + str(threshold)
#     dfg_base_dir = "./outputs/" + data_type + "/" + model_name + "/dfgs/" + t_path + "/"
#
#     start_idx = 0
#     end_idx = 10000
#     batch_size = 100
#     t_func = gt_scaled_t
#     output_layer_idx = len(model.layers) - 1
#     exclude_layers = nn_util.get_exclude_layers(model_json)
#     node_related_exclude_layers = nn_util.get_node_related_exclude_layers(model_json)
#
#     # 1. extract covered dfgs from contris
#     for idx in range(start_idx, end_idx, batch_size):
#         batch_start_idx = idx
#         batch_end_idx = idx + batch_size
#         post_fix = str(batch_start_idx) + "-" + str(batch_end_idx)
#         filename = base_dir + "/contris_batch_{0}_input_{1}.pkl".format(int(idx / batch_size), post_fix)
#         # filename = base_dir + "/contris_" + post_fix + ".pkl"
#         print("INFO, extracting and aggregating covered dfgs from image-contris {0} to {1}.".format(batch_start_idx,
#                                                                                                   batch_end_idx))
#         if data_type == 'mnist':
#             _, (x_test, _) = data_util.get_mnist_data()
#             x = x_test[batch_start_idx: batch_end_idx]
#         elif data_type == 'imagenet':
#             x = data_util.get_imgnet_test_data(st_idx=batch_start_idx, end_idx=batch_end_idx)
#         all_layers_out_vals = nn_contribution_util.get_all_layers_out_vals(model, x)
#
#         pred_labels, covered_dfgs = extract_covered_dfgs(model_json, filename, all_layers_out_vals, t_func, threshold)
#
#         covered_dfgs_file = dfg_base_dir + "dfg_" + str(batch_start_idx) + "-" + str(batch_end_idx) + ".pkl"
#         write_pkls_to_file(covered_dfgs, covered_dfgs_file, 'wb')
#
#     # 2. aggregate each category's covered dfgs, and compute each layer edge coverage of each category
#     each_category_covered_dfg_count_file = dfg_base_dir + "/categories/each_class_covered_dfg_count.pkl"
#
#     aggregated_all_categories_covered_dfg_path = dfg_base_dir + "/categories/"
#     if os.path.exists(aggregated_all_categories_covered_dfg_path):
#         shutil.rmtree(aggregated_all_categories_covered_dfg_path)
#     each_category_covered_dfg_count = {}
#     for idx in range(start_idx, end_idx, batch_size):
#         batch_start_idx = idx
#         batch_end_idx = idx + batch_size
#         covered_dfgs_file = dfg_base_dir + "dfg_" + str(batch_start_idx) + "-" + str(batch_end_idx) + ".pkl"
#         covered_dfgs = read_pkls_from_file(covered_dfgs_file)
#         for covered_dfg_idx in range(0, len(covered_dfgs)):
#             print("Aggregate each category's covered dfg over the {0}-th img.".format(idx + covered_dfg_idx))
#             covered_dfg = covered_dfgs[covered_dfg_idx]
#             category = get_category_by_covered_dfg(covered_dfg, output_shapes)
#             if category not in each_category_covered_dfg_count:
#                 each_category_covered_dfg_count[category] = 1
#             else:
#                 each_category_covered_dfg_count[category] += 1
#             aggregated_single_category_covered_dfg_file = dfg_base_dir + "/categories/aggregated_" + str(
#                 category) + "_dfg.pkl"
#             if os.path.exists(aggregated_single_category_covered_dfg_file):
#                 aggregated_single_category_covered_dfg = read_pkl_from_file(aggregated_single_category_covered_dfg_file)
#             else:
#                 aggregated_single_category_covered_dfg = {}
#             aggregated_single_category_covered_dfg = aggregate_graphs(aggregated_single_category_covered_dfg,
#                                                                       [covered_dfg])
#             write_pkl_to_file(aggregated_single_category_covered_dfg, aggregated_single_category_covered_dfg_file, 'wb')
#
#     write_pkl_to_file(each_category_covered_dfg_count, each_category_covered_dfg_count_file, 'wb')
#     print(each_category_covered_dfg_count)
#
#     all_ndcc_and_ecc = {}
#     each_layer_edges_num = count_each_layer_total_edges_num(model_json, input_shapes, output_shapes)
#     each_category_covered_dfg_count = read_pkl_from_file(each_category_covered_dfg_count_file)
#     for category in range(0, category_count):
#         all_ndcc_and_ecc[category] = {}
#         aggregated_single_category_covered_dfg_file = dfg_base_dir + "/categories/aggregated_" + str(
#             category) + "_dfg.pkl"
#         if os.path.exists(aggregated_single_category_covered_dfg_file):
#             aggregated_single_category_covered_dfg = read_pkl_from_file(aggregated_single_category_covered_dfg_file)
#             each_layer_covered_nodes_num, each_covered_covered_edges_num = calculate_each_layer_covered_node_and_edge(
#                 model,
#                 aggregated_single_category_covered_dfg,
#                 exclude_layers,
#                 node_related_exclude_layers)
#             each_layer_edges_num_of_one_class = copy.deepcopy(each_layer_edges_num)
#             each_layer_edges_num_of_one_class[output_layer_idx] = input_shapes[output_layer_idx][-1]
#             ndc_c, ndc_c_i, ec_c, ec_c_i = calculate_covered_ndc_and_ec_of_one_category(model,
#                                                                                         each_layer_covered_nodes_num,
#                                                                                         each_covered_covered_edges_num,
#                                                                                         each_layer_edges_num_of_one_class,
#                                                                                         exclude_layers,
#                                                                                         node_related_exclude_layers)
#             all_ndcc_and_ecc[category]['ndc_c'] = ndc_c
#             all_ndcc_and_ecc[category]['ndc_c_i'] = ndc_c_i
#             avg_ndc_c = np.mean(np.array(list(ndc_c_i.values())))
#             all_ndcc_and_ecc[category]['avg_ndc_c'] = avg_ndc_c
#
#             all_ndcc_and_ecc[category]['ec_c'] = ec_c
#             all_ndcc_and_ecc[category]['ec_c_i'] = ec_c_i
#             avg_ec_c = np.mean(np.array(list(ec_c_i.values())))
#             all_ndcc_and_ecc[category]['avg_ec_c'] = avg_ec_c
#
#             all_ndcc_and_ecc[category]['test_num'] = each_category_covered_dfg_count[category]
#     print("{0}_gt{1}_ndcc_and_ecc =".format(model_name, threshold), all_ndcc_and_ecc)
#     all_ndcc_and_ecc_file = dfg_base_dir + "all_ndcc_and_ecc.json"
#     write_json_to_file(all_ndcc_and_ecc, all_ndcc_and_ecc_file, 'w')
#
#     # 3. aggregate all covered dfgs, and compute each layer's edge coverage
#     output_batch_size = 100
#     aggregated_covered_dfg = {}
#     each_layer_edges_num = count_each_layer_total_edges_num(model_json, input_shapes, output_shapes)
#     all_ndc_and_ec = {}
#     for idx in range(start_idx, end_idx, batch_size):
#         batch_start_idx = idx
#         batch_end_idx = idx + batch_size
#         covered_dfgs_file = dfg_base_dir + "dfg_" + str(batch_start_idx) + "-" + str(batch_end_idx) + ".pkl"
#         covered_batch_dfgs = read_pkls_from_file(covered_dfgs_file)
#         aggregated_covered_dfg = aggregate_graphs(aggregated_covered_dfg, covered_batch_dfgs)
#         all_ndc_and_ec[batch_end_idx] = {}
#         if batch_end_idx % output_batch_size == 0:
#             # print("Computing edge coverage over the first {0} imgs".format(batch_end_idx))
#             aggregated_covered_dfg_file = dfg_base_dir + "/aggregated_covered_dfg/" + "dfg_" + str(
#                 batch_end_idx - output_batch_size) + "-" + str(batch_end_idx) + ".pkl"
#             write_pkl_to_file(aggregated_covered_dfg, aggregated_covered_dfg_file, 'wb')
#
#             aggregated_covered_dfg = read_pkl_from_file(aggregated_covered_dfg_file)
#             covered_nodes_num, covered_edges_num = calculate_each_layer_covered_node_and_edge(model,
#                                                                                               aggregated_covered_dfg,
#                                                                                               exclude_layers,
#                                                                                               node_related_exclude_layers)
#             ndc, ndc_i, ec, ec_i = calculate_covered_ndc_and_ec(model, covered_nodes_num, covered_edges_num,
#                                                                 each_layer_edges_num, exclude_layers,
#                                                                 node_related_exclude_layers)
#             all_ndc_and_ec[batch_end_idx]['ndc'] = ndc
#             all_ndc_and_ec[batch_end_idx]['ndc_i'] = ndc_i
#             avg_ndc = np.mean(np.array(list(ndc_i.values())))
#             all_ndc_and_ec[batch_end_idx]['avg_ndc'] = avg_ndc
#
#             all_ndc_and_ec[batch_end_idx]['ec'] = ec
#             all_ndc_and_ec[batch_end_idx]['ec_i'] = ec_i
#             avg_ec = np.mean(np.array(list(ec_i.values())))
#             all_ndc_and_ec[batch_end_idx]['avg_ec'] = avg_ec
#
#     print("{0}_gt{1}_ndc_and_ec =".format(model_name, threshold), all_ndc_and_ec)
#     all_ndc_and_ec_info_file = dfg_base_dir + "all_ndc_and_ec.json"
#     write_json_to_file(all_ndc_and_ec, all_ndc_and_ec_info_file, 'w')


def get_sliding_win_arr(model_json, model_name, sliding_win_width, stride):
    if model_name is None:
        model_name = model_json['config']['name']
    include_layers = get_included_layers(model_json, model_name)
    sliding_win_arr = []
    for idx_idx in range(len(include_layers) - 1, -1, -stride):
        if idx_idx + 1 - sliding_win_width < 0:
            break
        start_layer_idx = include_layers[max(idx_idx + 1 - sliding_win_width, 0)]
        end_layer_idx = include_layers[idx_idx]
        sliding_win_arr.append([start_layer_idx, end_layer_idx])
    return sliding_win_arr


def construct_complete_dfg(model):
    import nn_util
    print("INFO, constructing complete-dfg, this step may spend some seconds.")
    all_layers_outs_of_one_img = []
    for layer_idx in range(0, len(model.layers)):
        output_shape = list(model.layers[layer_idx].output_shape)
        output_shape.pop(0)
        all_layers_outs_of_one_img.append(np.ones(output_shape))

    model_json = json.loads(model.to_json())

    pred = [{} for i in range(0, len(model.layers))]
    input_shpaes = nn_util.get_dnn_each_layer_input_shape(model)
    for layer_idx in range(0, len(model.layers)):
        class_name = model_json["config"]["layers"][layer_idx]["class_name"]
        if class_name in DENSE_LAYERS + CONV_LAYERS:
            w = model.layers[layer_idx].get_weights()[0]
            pred[layer_idx]['c_i'] = [[] for i in range(0, w.shape[-1])]
            for n_idx in range(0, w.shape[-1]):
                # shape = list(input_shpaes[layer_idx]).pop(0)
                pred[layer_idx]['c_i'][n_idx] = np.ones(input_shpaes[layer_idx][-1])

    complete_dfg = {}
    end_layer = len(model.layers) - 1
    for n_idx in range(0, len(pred[end_layer]['c_i'])):
        pre_node = get_node_name(end_layer, n_idx)
        add_edge_to_graph(complete_dfg, pre_node, '')

    complete_dfg = extrac_one_covered_dfg(model_json, all_layers_outs_of_one_img, complete_dfg, pred, [],
                                          output_all_nodes)

    for n_idx in range(0, len(pred[end_layer]['c_i'])):
        pre_node = get_node_name(end_layer, n_idx)
        complete_dfg.pop(pre_node)
    print("INFO, finish complete-dfg construction.")
    return complete_dfg


def get_krps_of_k_sub_dfg(sub_graph, start_layer_idx, output_shapes):
    # import time
    all_k_reachable_path = {}
    # cnt = 0
    for out_node_idx in range(0, output_shapes[start_layer_idx][-1]):
        start_node = get_node_name(start_layer_idx, out_node_idx)
        if start_node not in sub_graph:
            continue
        print("INFO, get krps of k-win from node: {0}.".format(start_node))
        # st = time.time()
        all_k_reachable_path[start_node] = []
        stack = [[start_node, 0]]
        while stack:
            (v, next_child_idx) = stack[-1]
            if (v not in sub_graph) or (next_child_idx >= len(sub_graph[v])):
                if next_child_idx == 0:
                    krp = []
                    for node_idx in range(1, len(stack)):
                        # if get_layer_index_by_node_name(stack[node_idx][0]) in included_layer_arr:
                        krp.append(stack[node_idx][0])
                    krp = '-'.join(krp)
                    all_k_reachable_path[start_node].append(krp)
                stack.pop()
                continue
            next_child = sub_graph[v][next_child_idx]
            stack[-1][1] += 1
            stack.append([next_child, 0])
        # et = time.time()
        # print("dur.", et - st)
    return all_k_reachable_path


def get_pre_include_layers(model_json, include_layer):
    from nn_util import get_precursor_layer_idxes
    pre_layer_idxes = get_precursor_layer_idxes(model_json, include_layer)
    pre_include_layers = []
    for pre_layer_idx in pre_layer_idxes:
        pre_class_name = model_json["config"]["layers"][pre_layer_idx]["class_name"]
        while pre_class_name not in CONV_LAYERS + MULYIPY_TO_ONE_LAYERS + DENSE_LAYERS + INPUT_LAYER:
            pre_layer_idx = get_precursor_layer_idxes(model_json, pre_layer_idx)[0]
            pre_class_name = model_json["config"]["layers"][pre_layer_idx]["class_name"]
        pre_include_layers.append(pre_layer_idx)
    return pre_include_layers


def preprocess_dfg(model_json, model_name, dfg, output_shapes):
    from nn_util import get_precursor_layer_idxes
    included_layers = get_included_layers(model_json, model_name)
    new_dfg = {}
    for idx_idx in range(1, len(included_layers)):
        cur_included_layers = included_layers[idx_idx]
        for cur_included_layer in cur_included_layers:
            pre_layer_idxes = get_precursor_layer_idxes(model_json, cur_included_layer)
            if len(pre_layer_idxes) == 1:
                pre_class_name = model_json["config"]["layers"][pre_layer_idxes[0]]["class_name"]
                if pre_class_name in FLATTEN_LAYERS:
                    pre_layer_idxes = get_precursor_layer_idxes(model_json, pre_layer_idxes[0])
            pre_included_layers = get_pre_include_layers(model_json, cur_included_layer)
            for pre_idx_idx in range(0, len(pre_layer_idxes)):
                pre_layer_idx = pre_layer_idxes[pre_idx_idx]
                pre_included_layer_idx = pre_included_layers[pre_idx_idx]
                for pre_node_idx in range(0, output_shapes[pre_layer_idx][-1]):
                    pre_layer_node = get_node_name(pre_layer_idx, pre_node_idx)
                    if pre_layer_node in dfg:
                        pre_include_layer_node = get_node_name(pre_included_layer_idx, pre_node_idx)
                        # if pre_include_layer_node not in dfg:
                        #     print(pre_layer_node, dfg[pre_layer_node])
                        #     print("ERROR，")
                        new_dfg[pre_include_layer_node] = dfg[pre_layer_node]
    return new_dfg


def reduce_uncoved_krps(uncoved_krps, coved_k_dfg):
    for start_node in uncoved_krps:
        if start_node not in coved_k_dfg:
            continue
        if len(uncoved_krps[start_node]) == len(coved_k_dfg[start_node]):
            uncoved_krps[start_node].clear()
        for uncoved_path_idx in range(len(uncoved_krps[start_node]) - 1, -1, -1):
            uncoved_path = uncoved_krps[start_node][uncoved_path_idx]
            uncoved_path = uncoved_path.split('-')
            for uncoved_n_idx in range(0, len(uncoved_path) - 1):
                pre_node = uncoved_path[uncoved_n_idx]
                post_node = uncoved_path[uncoved_n_idx + 1]
                if pre_node not in coved_k_dfg or post_node not in coved_k_dfg[pre_node]:
                    continue
                uncoved_krps[start_node].pop(uncoved_path_idx)
    return uncoved_krps


def get_krps_of_k_sub_dfg_in_parallel(sub_graph, start_layer_idx, output_shapes):
    all_k_reachable_path = {}

    pool = multiprocessing.Pool()
    rets = []
    for out_node_idx in range(0, output_shapes[start_layer_idx][-1]):
        start_node = get_node_name(start_layer_idx, out_node_idx)
        if start_node not in sub_graph:
            continue
        print("INFO, get krps of k-win from node: {0}.".format(start_node))
        # take a tree from sub_graph start at start_node
        tree = sub_graph.copy()
        for tmp_out_node_idx in range(0, output_shapes[start_layer_idx][-1]):
            tmp_node = get_node_name(start_layer_idx, tmp_out_node_idx)
            if start_node not in sub_graph:
                continue
            if tmp_node != start_node:
                tree.pop(tmp_node)

        ret = pool.apply_async(get_krps_of_k_sub_dfg, (tree, start_layer_idx, output_shapes))
        rets.append(ret)
    pool.close()
    pool.join()
    for ret in rets:
        tmp_ret = ret.get()
        for key in tmp_ret:
            all_k_reachable_path[key] = tmp_ret[key]
    return all_k_reachable_path


def write_tree_krps(tree, tree_krp_file, start_layer_idx, output_shapes):
    tree_krps = get_krps_of_k_sub_dfg(tree, start_layer_idx, output_shapes)
    write_pkl_to_file(tree_krps, tree_krp_file, 'wb')
    return tree_krps


def write_krps_of_trees(sub_graph, start_layer_idx, output_shapes, krps_path):
    cpu_core_num = multiprocessing.cpu_count() * 4
    for out_n_idx in range(0, output_shapes[start_layer_idx][-1], cpu_core_num):
        tmp_end_n_idx = min(out_n_idx + cpu_core_num, output_shapes[start_layer_idx][-1])
        pool = multiprocessing.Pool()
        for tmp_n_idx in range(out_n_idx, tmp_end_n_idx):
            start_node = get_node_name(start_layer_idx, tmp_n_idx)
            tree_krp_file = krps_path + '/krps_' + start_node + '.pkl'
            print("INFO, async write {0} started tree's krps into {1}".format(start_node, tree_krp_file))
            # take a tree from sub_graph start at start_node
            tree = sub_graph.copy()
            for tmp_out_node_idx in range(0, output_shapes[start_layer_idx][-1]):
                tmp_node = get_node_name(start_layer_idx, tmp_out_node_idx)
                if start_node not in sub_graph:
                    continue
                if tmp_node != start_node:
                    tree.pop(tmp_node)
            pool.apply_async(write_tree_krps, (tree, tree_krp_file, start_layer_idx, output_shapes))
        pool.close()
        pool.join()


def processed_dfgs(model_json, model_name, output_shapes, dfg_base_dir, start_dfg_idx, end_dfg_idx, batch_size):
    print("INFO, start the pre-process of {0}'s dfgs".format(model_name))
    for coved_dfg_idx in range(start_dfg_idx, end_dfg_idx, batch_size):
        batch_postfix = str(coved_dfg_idx) + "-" + str(coved_dfg_idx + batch_size)
        processed_coved_dfgs_file = dfg_base_dir + "/processed_dfgs/processed_dfg_" + batch_postfix + ".pkl"
        if os.path.exists(processed_coved_dfgs_file):
            continue
        coved_dfgs_file = dfg_base_dir + "/dfg_" + batch_postfix + ".pkl"
        coved_dfgs = read_pkls_from_file(coved_dfgs_file)
        processed_coved_dfgs = []
        for batch_coved_dfg_idx in range(0, len(coved_dfgs)):
            covered_dfg = coved_dfgs[batch_coved_dfg_idx]
            processed_covered_dfg = preprocess_dfg(model_json, model_name, covered_dfg, output_shapes)
            processed_coved_dfgs.append(processed_covered_dfg)
        write_pkls_to_file(processed_coved_dfgs, processed_coved_dfgs_file, 'wb')
    print("INFO, finish the pre-process of {0}'s dfgs".format(model_name))


def mean_by_last_axis(arr):
    mean_vals = []
    for idx in range(0, arr.shape[-1]):
        val = np.mean(arr[..., idx])
        mean_vals.append(val)
    return np.array(mean_vals)


def arg_max_flags(vals):
    # flags = np.zeros(len(vals), dtype=int)
    flags = [0 for i in range(0, len(vals))]
    arg_max_idx = np.argmax(vals)
    flags[arg_max_idx] = 1
    return flags


def compute_neuron_states_from_layer_outs(model, all_layers_out_vals, t_func, *args):
    layers_count = len(model.layers)
    neuron_states = []
    for layer_idx in range(0, layers_count):
        layer_neuron_count = model.layers[layer_idx].output_shape[-1]
        layer_neuron_states = np.zeros(layer_neuron_count, dtype=int)
        for img_idx in range(0, len(all_layers_out_vals[layer_idx])):
            print("processing {0}-th image in {1}-th layer".format(img_idx, layer_idx))
            layer_output_of_single_img = all_layers_out_vals[layer_idx][img_idx]
            layer_output_of_single_img = mean_by_last_axis(layer_output_of_single_img)
            if layer_idx == layers_count - 1:
                neuron_activated_flags = np.array(arg_max_flags(layer_output_of_single_img))
            else:
                neuron_activated_flags = np.array(t_func(layer_output_of_single_img, args))
            layer_neuron_states += neuron_activated_flags
        neuron_states.append(layer_neuron_states)
    return neuron_states


def compute_neuron_states(model, imgs, t_func, *args):
    from nn_util import get_all_layers_out_vals
    all_layers_out_vals = get_all_layers_out_vals(model, imgs)
    return compute_neuron_states_from_layer_outs(model, all_layers_out_vals, t_func, args)


def nc_main():
    import data_util
    import copy
    from keras.models import load_model
    from keras.models import Model
    from keras.applications.vgg19 import VGG19
    from keras.applications.resnet50 import ResNet50
    import nn_util

    # model_name = 'lenet1'
    # model_name = 'lenet4'
    # model_name = 'lenet5'
    # model_path = './models/{0}-relu'.format(model_name)
    # model = load_model(model_path)
    # data_type = 'mnist'

    model = ResNet50(weights='imagenet')
    model_name = 'resnet50'
    # model = VGG19(weights='imagenet')
    # model_name = 'vgg19'
    data_type = 'imagenet'

    t_func = gte_scaled_t
    threshold = 0
    t_path = "gt_" + str(threshold)
    nc_base_dir = "./outputs/" + data_type + "/" + model_name + "/ncs/" + t_path + "/"

    start_idx = 0
    end_idx = 5000
    batch_size = 100
    model_json = json.loads(model.to_json())
    node_related_exclude_layers = nn_util.get_node_related_exclude_layers(model_json)
    all_covered_neuron_states_file = nc_base_dir + "/all_covered_neuron_states.pkl"

    all_covered_neuron_states = {}
    covered_neuron_state = None
    for idx in range(start_idx, end_idx, batch_size):
        batch_start_idx = idx
        batch_end_idx = idx + batch_size
        post_fix = str(batch_start_idx) + "-" + str(batch_end_idx)
        print("Computing neurons' activation states over contris {0}.".format(post_fix))
        if data_type == 'mnist':
            _, (x_test, _) = data_util.get_mnist_data()
            x = x_test[batch_start_idx: batch_end_idx]
        elif data_type == 'imagenet':
            x = data_util.get_imgnet_test_data(st_idx=batch_start_idx, end_idx=batch_end_idx)
        neuron_states = compute_neuron_states(model, x, t_func, threshold)
        neuron_states_file = nc_base_dir + "/" + post_fix + ".pkl"
        write_pkl_to_file(neuron_states, neuron_states_file, 'wb')

        if not covered_neuron_state:
            covered_neuron_state = neuron_states
        else:
            for layer_idx in range(0, len(model.layers)):
                covered_neuron_state[layer_idx] += neuron_states[layer_idx]
        all_covered_neuron_states[batch_end_idx] = copy.deepcopy(covered_neuron_state)
    write_pkl_to_file(all_covered_neuron_states, all_covered_neuron_states_file, 'wb')

    all_covered_nc = {}
    all_covered_neuron_states = read_pkl_from_file(all_covered_neuron_states_file)
    for batch_idx in all_covered_neuron_states:
        all_covered_nc[batch_idx] = {}
        covered_neuron_states = all_covered_neuron_states[batch_idx]
        nc_i = {}
        dnn_covered_neurons_count = 0
        dnn_neurons_count = 0
        for layer_idx in range(0, len(covered_neuron_states)):
            if layer_idx in node_related_exclude_layers:
                continue
            layer_covered_neuron_count = np.sum(np.clip(covered_neuron_states[layer_idx], 0, 1))
            layer_neuron_count = len(covered_neuron_states[layer_idx])
            nc_i[layer_idx] = layer_covered_neuron_count / layer_neuron_count
            dnn_covered_neurons_count += layer_covered_neuron_count
            dnn_neurons_count += layer_neuron_count
        all_covered_nc[batch_idx]['nc_i'] = nc_i
        all_covered_nc[batch_idx]['nc'] = dnn_covered_neurons_count / dnn_neurons_count
        avg_nc = np.mean(np.array(list(nc_i.values())))
        all_covered_nc[batch_idx]['avg_nc'] = avg_nc

    print("{0}_gt{1}_nc =".format(model_name, threshold), all_covered_nc)
    all_covered_nc_file = nc_base_dir + "all_nc.json"
    write_json_to_file(all_covered_nc_file, all_covered_nc_file, 'w')


def caculate_similarity_of_two_dfgs(model, fst_dfg, scd_dfg, exclude_layers=[], exclude_node_related_layers=[]):
    overlap_dfg = {}
    overlap_start_nodes = set(fst_dfg.keys()) & set(scd_dfg.keys())
    overlap_edges_count = 0
    for start_node in overlap_start_nodes:
        connected_nodes_set_in_fst_dfg = set(fst_dfg[start_node])
        connected_nodes_set_in_scd_dfg = set(scd_dfg[start_node])
        connected_overlap_nodes = connected_nodes_set_in_fst_dfg & connected_nodes_set_in_scd_dfg
        overlap_dfg[start_node] = connected_overlap_nodes
        overlap_edges_count += len(connected_overlap_nodes)

    union_dfg = {}
    union_start_nodes = set(fst_dfg.keys()) | set(scd_dfg.keys())
    union_edges_count = 0
    for start_node in union_start_nodes:
        connected_nodes_set_in_fst_dfg = set()
        connected_nodes_set_in_scd_dfg = set()
        if start_node in fst_dfg:
            connected_nodes_set_in_fst_dfg = set(fst_dfg[start_node])
        if start_node in scd_dfg:
            connected_nodes_set_in_scd_dfg = set(scd_dfg[start_node])
        connected_union_nodes = connected_nodes_set_in_fst_dfg | connected_nodes_set_in_scd_dfg
        union_dfg[start_node] = connected_union_nodes
        union_edges_count += len(connected_union_nodes)

    each_layer_similarity = {}
    _, each_layer_overlap_edges_num = calculate_each_layer_covered_node_and_edge(model, overlap_dfg, exclude_layers,
                                                                                 exclude_node_related_layers)
    _, each_layer_union_edges_num = calculate_each_layer_covered_node_and_edge(model, union_dfg, exclude_layers,
                                                                               exclude_node_related_layers)
    for layer_idx in each_layer_overlap_edges_num:
        each_layer_similarity[layer_idx] = each_layer_overlap_edges_num[layer_idx] / each_layer_union_edges_num[
            layer_idx]

    dfg_similarity = overlap_edges_count / union_edges_count
    return dfg_similarity, each_layer_similarity


def caculate_avg_similarity_of_each_layer(category_count, similaries):
    similarities_of_each_layer = {}
    for category in range(0, category_count):
        for other_category in range(category + 1, category_count):
            each_layer_similarity = similaries[category][other_category]['smly_i']
            for layer_idx in each_layer_similarity:
                if layer_idx not in similarities_of_each_layer:
                    similarities_of_each_layer[layer_idx] = [each_layer_similarity[layer_idx]]
                else:
                    similarities_of_each_layer[layer_idx].append(each_layer_similarity[layer_idx])
    similarity_of_each_layer = {}
    for layer_idx in similarities_of_each_layer:
        similarity_of_each_layer[layer_idx] = {}
        similarities_of_curlayer = similarities_of_each_layer[layer_idx]
        similarity_of_each_layer[layer_idx]['mean'] = np.mean(similarities_of_curlayer)
        similarity_of_each_layer[layer_idx]['var'] = np.var(similarities_of_curlayer)
        similarity_of_each_layer[layer_idx]['std'] = np.std(similarities_of_curlayer)
    return similarity_of_each_layer


# def caculate_similarity_of_all_categories():
#     from keras.models import load_model
#     from keras.models import Model
#     from keras.applications.vgg19 import VGG19
#     from keras.applications.resnet50 import ResNet50
#     import nn_util
#
#     # # lenet1
#     # model_path = './models/lenet1-relu'
#     # model_name = 'lenet1'
#     # lenet4
#     # model_path = './models/lenet4-relu'
#     # model_name = 'lenet4'
#     # lenet5
#     model_path = './models/lenet5-relu'
#     model_name = 'lenet5'
#
#     model = load_model(model_path)
#     data_type = 'mnist'
#     category_count = 10
#
#     # model = ResNet50(weights='imagenet')
#     # model_name = 'resnet50'
#     # # model = VGG19(weights='imagenet')
#     # # model_name = 'vgg19'
#     # data_type = 'imagenet'
#     # category_count = 1000
#
#     threshold = 0.75
#     t_path = "gt_" + str(threshold)
#     dfg_base_dir = "./outputs/" + data_type + "/" + model_name + "/dfgs/" + t_path
#     model_json = json.loads(model.to_json())
#     exclude_layers = nn_util.get_exclude_layers(model_json)
#     node_related_exclude_layers = nn_util.get_node_related_exclude_layers(model_json)
#     similaries = {}
#     for category in range(0, category_count):
#         similaries[category] = {}
#         cur_c_dfg_file = dfg_base_dir + "/categories/aggregated_" + str(category) + "_dfg.pkl"
#         cur_c_dfg = read_pkl_from_file(cur_c_dfg_file)
#         for other_category in range(0, category_count):
#             if category == other_category:
#                 continue
#             similaries[category][other_category] = {}
#             oth_c_dfg_file = dfg_base_dir + "/categories/aggregated_" + str(other_category) + "_dfg.pkl"
#             oth_c_dfg = read_pkl_from_file(oth_c_dfg_file)
#             dfg_similarity, each_layer_similarity = caculate_similarity_of_two_dfgs(model, cur_c_dfg, oth_c_dfg,
#                                                                                     exclude_layers,
#                                                                                     node_related_exclude_layers)
#             similaries[category][other_category]['smly'] = dfg_similarity
#             similaries[category][other_category]['smly_i'] = each_layer_similarity
#     similaries['avg_smly_i'] = caculate_avg_similarity_of_each_layer(category_count, similaries)
#     print("{0}_gt{1}_edge_similarity = {2}".format(model_name, threshold, similaries))
#     similaries_file = dfg_base_dir + "/categories/all_categories_similaries.json"
#     write_json_to_file(similaries, similaries_file, 'w')


def extract_dfgs_and_compute_conc(model, st_idx=0, end_idx=10000, contris_batch_size=100, threshold=0):
    import data_util
    import nn_util
    model_name = model.name
    if model_name in MNIST_MODELS:
        data_type='mnist'
    elif model_name in IMAGENET_MODELS:
        data_type = 'imagenet'
    else:
        raise ValueError("Fail to auto claer session, the {0} is not support!".format(model_name))

    model_json = json.loads(model.to_json())
    output_shapes = nn_util.get_dnn_each_layer_output_shape(model)
    input_shapes = nn_util.get_dnn_each_layer_input_shape(model)
    base_dir = "./outputs/" + data_type + "/" + model_name + "/contris/"
    t_path = "gt_" + str(threshold)
    dfg_base_dir = "./outputs/" + data_type + "/" + model_name + "/dfgs/" + t_path + "/"

    output_batch_size = 100
    t_func = gt_scaled_t
    exclude_layers = nn_util.get_exclude_layers(model_json)
    node_related_exclude_layers = nn_util.get_node_related_exclude_layers(model_json)
    all_pre_related_layers = nn_util.get_all_pre_related_layers(model_json)

    # 1. extract covered dfgs from contris
    for output_idx in range(st_idx, end_idx, output_batch_size):
        output_batch_start_idx = output_idx
        output_batch_end_idx = output_idx + output_batch_size
        output_batch_dfg = {}
        for contris_batch_idx in range(output_batch_start_idx, output_batch_end_idx, contris_batch_size):
            contris_start_idx = contris_batch_idx
            contris_end_idx = contris_batch_idx + contris_batch_size
            contris_post_fix = str(contris_start_idx) + "-" + str(contris_end_idx)
            filename = base_dir + "/contris_" + contris_post_fix + ".pkl"
            print("INFO, extract and aggregate dfgs (with t>{2}) from contris {0} to {1}.".format(contris_start_idx, contris_end_idx, threshold))
            if data_type == 'mnist':
                _, (x_test, _) = data_util.get_mnist_data()
                x = x_test[contris_start_idx: contris_end_idx]
            elif data_type == 'imagenet':
                x = data_util.get_imgnet_test_data(st_idx=contris_start_idx, end_idx=contris_end_idx)
            all_layers_out_vals = nn_util.get_all_layers_out_vals(model, x)

            pred_labels, covered_dfgs = extract_covered_dfgs_from_contris_file(model_json, filename, all_layers_out_vals, t_func, threshold)
            output_batch_dfg = aggregate_graphs(output_batch_dfg, covered_dfgs)

        output_batch_dfg_file = dfg_base_dir + "dfg_" + str(output_batch_start_idx) + "-" + str(output_batch_end_idx) + ".pkl"
        write_pkl_to_file(output_batch_dfg, output_batch_dfg_file, 'wb')

    # 2. aggregate all covered dfgs, and compute each layer's edge coverage
    aggregated_covered_dfg = {}
    total_covered_dfg_file = dfg_base_dir + "total_covered_dfg.pkl"
    each_layer_edges_num = count_each_layer_total_edges_num(model_json, input_shapes, output_shapes)
    all_ndc_and_ec = {}
    for output_idx in range(st_idx, end_idx, output_batch_size):
        output_batch_start_idx = output_idx
        output_batch_end_idx = output_idx + output_batch_size
        batch_covered_dfg_file = dfg_base_dir + "dfg_" + str(output_batch_start_idx) + "-" + str(
            output_batch_end_idx) + ".pkl"
        batch_covered_dfg = read_pkl_from_file(batch_covered_dfg_file)
        aggregated_covered_dfg = aggregate_graphs(aggregated_covered_dfg, [batch_covered_dfg])
        all_ndc_and_ec[output_batch_end_idx] = {}

        print("Computing edge coverage over the first {0} imgs".format(output_batch_end_idx))
        covered_nodes_num, covered_edges_num = calculate_each_layer_covered_node_and_edge(model, aggregated_covered_dfg, exclude_layers, node_related_exclude_layers, all_pre_related_layers)
        ndc, ndc_i, ec, ec_i = calculate_covered_ndc_and_ec(model, covered_nodes_num, covered_edges_num, each_layer_edges_num, exclude_layers, node_related_exclude_layers)
        all_ndc_and_ec[output_batch_end_idx]['ndc'] = ndc
        all_ndc_and_ec[output_batch_end_idx]['ndc_i'] = ndc_i
        avg_ndc = np.mean(np.array(list(ndc_i.values())))
        all_ndc_and_ec[output_batch_end_idx]['avg_ndc'] = avg_ndc

        all_ndc_and_ec[output_batch_end_idx]['ec'] = ec
        all_ndc_and_ec[output_batch_end_idx]['ec_i'] = ec_i
        avg_ec = np.mean(np.array(list(ec_i.values())))
        all_ndc_and_ec[output_batch_end_idx]['avg_ec'] = avg_ec
    # print("{0}_gt{1}_ndcc_and_ecc =".format(model_name, threshold), all_ndcc_and_ecc)
    print("{0}_gt{1}_ndc_and_ec =".format(model_name, threshold), all_ndc_and_ec)
    all_ndc_and_ec_info_file = dfg_base_dir + "all_ndc_and_ec.json"
    write_json_to_file(all_ndc_and_ec, all_ndc_and_ec_info_file, 'w')
    write_pkl_to_file(aggregated_covered_dfg, total_covered_dfg_file, 'wb')



# def merger_each_category_dfg():
#     # model_name = 'vgg19'
#     model_name = 'resnet50'
#     data_type = 'imagenet'
#     threshold = 0
#     t_path = "gt_" + str(threshold)
#     dfg_base_dir = "./outputs/" + data_type + "/" + model_name + "/dfgs/" + t_path + "/"
#     category_count = 1000
#     tmp_c_dfg_files = ['dfgs_0-5000', 'dfgs_5000-10000', 'dfgs_10000-15000', 'dfgs_15000-20000']
#     tmp_c_dfg_base_dir = "./outputs/" + data_type + "/" + model_name + "/"
#     for category in range(0, category_count):
#         c_dfg_file = dfg_base_dir + "/categories/aggregated_" + str(category) + "_dfg.pkl"
#         c_dfg = {}
#         for tmp_c_dfg_file in tmp_c_dfg_files:
#             batch_c_dfg_file = tmp_c_dfg_base_dir + "/" + tmp_c_dfg_file + "/" + t_path + "/categories/aggregated_" + str(
#                 category) + "_dfg.pkl"
#             if not os.path.exists(batch_c_dfg_file):
#                 continue
#             batch_c_dfg = read_pkl_from_file(batch_c_dfg_file)
#             print("aggregate dfg from", batch_c_dfg_file)
#             c_dfg = aggregate_graphs(c_dfg, [batch_c_dfg])
#         write_pkl_to_file(c_dfg, c_dfg_file, 'wb')
#
#     each_category_covered_dfg_count_file = dfg_base_dir + "/categories/each_class_covered_dfg_count.pkl"
#     each_category_covered_dfg_count = {}
#     for tmp_c_dfg_file in tmp_c_dfg_files:
#         tmp_c_dfg_count_file = tmp_c_dfg_base_dir + "/" + tmp_c_dfg_file + "/" + t_path + "/categories/each_class_covered_dfg_count.pkl"
#         tmp_c_dfg_count = read_pkl_from_file(tmp_c_dfg_count_file)
#         for category in range(0, category_count):
#             if category not in tmp_c_dfg_count:
#                 continue
#             if category not in each_category_covered_dfg_count:
#                 each_category_covered_dfg_count[category] = tmp_c_dfg_count[category]
#             else:
#                 each_category_covered_dfg_count[category] += tmp_c_dfg_count[category]
#     print(each_category_covered_dfg_count)
#     write_pkl_to_file(each_category_covered_dfg_count, each_category_covered_dfg_count_file, 'wb')



if __name__ == '__main__':
    import argparse
    import keras
    from keras.applications.vgg19 import VGG19
    from keras.applications.resnet50 import ResNet50
    from keras import backend as K
    from keras.models import load_model

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model_name', '-mn', type=str, default='lenet1')
    parser.add_argument('--start_index', '-st', type=int, default=0)
    parser.add_argument('--end_index', '-ed', type=int, default=10000)
    parser.add_argument('--batch_size', '-bz', type=int, default=100)
    parser.add_argument('--threshold', '-t', type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    if model_name == 'vgg19':
        model = VGG19(weights='imagenet', backend=K, layers=keras.layers, models=keras.models, utils=keras.utils)
    elif model_name == 'resnet50':
        model = ResNet50(weights='imagenet', backend=K, layers=keras.layers, models=keras.models, utils=keras.utils)
    elif model_name in ['lenet1', 'lenet4', 'lenet5']:
        model_path = './models/{0}-relu'.format(model_name)
        model = load_model(model_path)
        model.name = model_name
    else:
        raise ValueError("Fail to auto claer session, the {0} is not support!".format(model_name))


    st_idx = int(args.start_index)
    end_idx = int(args.end_index)
    batch_size = int(args.batch_size)
    t = float(args.threshold)
    print("model name: {0}, start index: {1}, end index: {2}, batch size: {3}, threshold: {4}".format(model_name, st_idx, end_idx, batch_size, t))
    extract_dfgs_and_compute_conc(model, st_idx, end_idx, batch_size, t)
