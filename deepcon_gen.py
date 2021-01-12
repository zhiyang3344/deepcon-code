import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications import imagenet_utils
from keras import backend as K
from PIL import Image
from keras.models import load_model
import data_util
import tensorflow as tf
import time
import dfg_util
import nn_util
import nn_contribution_util
from ioutil import *
import shutil
import random
import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DENSE_LAYERS = ['Dense']
CONV_LAYERS = ['Conv2D']
FLATTEN_LAYERS = ["Flatten"]


def get_pre_activation(model, node_name, with_bias_flag=False):
    model_json = json.loads(model.to_json())
    layer_idx = dfg_util.get_layer_index_by_node_name(node_name)
    class_name = model_json["config"]["layers"][layer_idx]["class_name"]
    config_in_layer = model_json["config"]["layers"][layer_idx]["config"]

    def get_conv_neuron_prenon(model, node_name, with_bias_flag=False):
        layer_idx = dfg_util.get_layer_index_by_node_name(node_name)
        node_idx = dfg_util.get_node_idx_by_node_name(node_name)
        weights = model.layers[layer_idx].get_weights()
        layer_input = model.layers[layer_idx].input

        strides = tuple(model_json["config"]["layers"][layer_idx]["config"]["strides"])
        padding = model_json["config"]["layers"][layer_idx]["config"]["padding"]
        data_format = model_json["config"]["layers"][layer_idx]["config"]["data_format"]
        dilation_rate = tuple(model_json["config"]["layers"][layer_idx]["config"]["dilation_rate"])
        kernel = weights[0]
        kernel_of_node = kernel[..., node_idx:node_idx + 1]
        bias = weights[1][..., node_idx:node_idx + 1]
        cnov2d = K.conv2d(layer_input, kernel_of_node, strides, padding, data_format, dilation_rate)
        if with_bias_flag:
            pre_activation = cnov2d[..., 0] + bias
        else:
            pre_activation = cnov2d[..., 0]
        return pre_activation

    def get_dense_neuron_prenon(model, node_name, with_bias_flag=False):
        layer_idx = dfg_util.get_layer_index_by_node_name(node_name)
        node_idx = dfg_util.get_node_idx_by_node_name(node_name)
        weights = model.layers[layer_idx].get_weights()
        layer_input = model.layers[layer_idx].input
        w = weights[0]
        b = weights[1]
        w_of_node = w[..., node_idx:node_idx + 1]
        if np.any(w_of_node == 0.0):
            print('WARNING, the weights contain zero')
        b_of_node = b[..., node_idx]
        dot = K.dot(layer_input, K.constant(w_of_node))
        if with_bias_flag:
            pre_activation = dot + b_of_node
        else:
            pre_activation = dot
        pre_activation = pre_activation[..., 0]
        return pre_activation

    def get_activation_neuron_prenon(model, node_name):
        layer_idx = dfg_util.get_layer_index_by_node_name(node_name)
        node_idx = dfg_util.get_node_idx_by_node_name(node_name)
        layer_input = model.layers[layer_idx].input
        return layer_input[..., node_idx]

    def get_flatten_neuron_prenon(model, node_name):
        layer_idx = dfg_util.get_layer_index_by_node_name(node_name)
        node_idx = dfg_util.get_node_idx_by_node_name(node_name)
        pre_layer_index = nn_contribution_util.get_precursor_layer_idxes(model_json, layer_idx)[0]
        pre_class_name = model_json["config"]["layers"][pre_layer_index]["class_name"]
        if pre_class_name == 'MaxPooling2D':
            pre_layer_output_shape = model.layers[pre_layer_index].output_shape
            pre_layer_output_shape = (pre_layer_output_shape[1], pre_layer_output_shape[2], pre_layer_output_shape[3])
            dense2featuremap_mapper = dfg_util.get_dense2featuremap_mapper(pre_layer_output_shape)
            h_idx, w_idx, fm_idx = dense2featuremap_mapper[node_idx]

            pool_size = tuple(model_json["config"]["layers"][pre_layer_index]["config"]["pool_size"])
            padding = model_json["config"]["layers"][pre_layer_index]["config"]["padding"]
            strides = tuple(model_json["config"]["layers"][pre_layer_index]["config"]["strides"])
            data_format = model_json["config"]["layers"][pre_layer_index]["config"]["data_format"]

            pre_pre_layer_index = nn_contribution_util.get_precursor_layer_idxes(model_json, pre_layer_index)[0]
            pre_pre_class_name = model_json["config"]["layers"][pre_pre_layer_index]["class_name"]
            if pre_pre_class_name == 'Conv2D':
                pre_pre_node_name = dfg_util.get_node_name(pre_pre_layer_index, fm_idx)
                pre_pre_activation = get_conv_neuron_prenon(model, pre_pre_node_name)
                tensor_in_maxpooling2d = K.pool2d(K.expand_dims(pre_pre_activation), pool_size, strides, padding,
                                                  data_format)
                pre_activation_in_maxpooling2d = tensor_in_maxpooling2d[..., 0][:, h_idx, w_idx]
                return pre_activation_in_maxpooling2d
        return model.layers[layer_idx].output[..., node_idx]

    if 'activation' in config_in_layer and config_in_layer['activation'] == 'relu':
        if class_name == "Conv2D":
            pre_activation = get_conv_neuron_prenon(model, node_name, with_bias_flag)
        elif class_name == "Dense":
            pre_activation = get_dense_neuron_prenon(model, node_name, with_bias_flag)
        elif class_name == "Activation":
            pre_activation = get_activation_neuron_prenon(model, node_name)
        else:
            raise AssertionError("ERROR, I can't handle the type of layer: " + str(class_name))
    elif class_name == 'Flatten':
        pre_activation = get_flatten_neuron_prenon(model, node_name)
    else:
        layer_idx = dfg_util.get_layer_index_by_node_name(node_name)
        node_idx = dfg_util.get_node_idx_by_node_name(node_name)
        weighted_layer_idx = nn_util.get_pre_weighted_layers(model, layer_idx)
        pre_activation = model.layers[weighted_layer_idx].output[..., node_idx]
    return pre_activation


def scale(arr):
    min_val = min(arr)
    max_val = max(arr)
    scaled_arr = []
    for val in arr:
        if val == 0.0:
            scaled_arr.append(scale_val)
        else:
            scale_val = (val - min_val) / (max_val - min_val)
            scaled_arr.append(scale_val)
    return np.array(scaled_arr)


def get_scaled_val(val, arr):
    if val == 0.0:
        return val
    min_val = min(arr)
    max_val = max(arr)
    scale_val = (val - min_val) / (max_val - min_val)
    return scale_val


def mean_by_last_axis(arr):
    mean_vals = []
    for idx in range(0, len(arr)):
        val = np.mean(arr[..., idx])
        mean_vals.append(val)
    return np.array(mean_vals)


# Note, call this function carefully, it's excution duration may take several seconds.
def get_sub_covered_dfg(model, x, node, t_func, *args):
    exclude_layers = []
    end_layer_idx = dfg_util.get_layer_index_by_node_name(node)
    for layer_idx in range(0, end_layer_idx):
        exclude_layers.append(layer_idx)
    sub_covered_dfg = dfg_util.get_covered_dfg_of_one_img(model, x, exclude_layers, t_func, *args)
    return sub_covered_dfg


def get_allowed_image_changes(model_name, original_image):
    if model_name in ('vgg19', 'resnet50'):
        gap = 255.0
    elif model_name in ('lenet1', 'lenet4', 'lenet5'):
        gap = 1.0
    dis_factor = 0.3
    max_change_above = original_image + dis_factor * gap
    max_change_below = original_image - dis_factor * gap
    return max_change_below, max_change_above


def clip_image(model_name, max_change_below, max_change_above, hacked_image):
    hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
    if model_name in ('vgg', 'resnet'):
        hacked_image[0][:, :, 0] = np.clip(hacked_image[0][:, :, 0], 0.0 - 103.939, 255.0 - 103.939)
        hacked_image[0][:, :, 1] = np.clip(hacked_image[0][:, :, 1], 0.0 - 116.779, 255.0 - 116.779)
        hacked_image[0][:, :, 2] = np.clip(hacked_image[0][:, :, 2], 0.0 - 123.680, 255.0 - 123.680)
    else:
        hacked_image[0] = np.clip(hacked_image[0], 0.0, 1.0)
    return hacked_image


def is_neuron_activated(cur_neuron_idx, layer_output_vals, t_func=dfg_util.gt_scaled_t, *args):
    each_neuron_means_in_layer = []
    if layer_output_vals.ndim > 1:
        for neuron_idx in range(0, layer_output_vals.shape[-1]):
            each_neuron_means_in_layer.append(np.mean(layer_output_vals[..., neuron_idx]))
    else:
        each_neuron_means_in_layer = layer_output_vals
    neurons_activations_flags = t_func(each_neuron_means_in_layer, args[0])
    return neurons_activations_flags[cur_neuron_idx]


def activate_neuron(model, original_image, neuron, model_name, esp=0.05, MAX_ITERATIONS=1000, t_func=dfg_util.gt_scaled_t, *args):
    model_input_layer = model.layers[0].input
    max_change_below, max_change_above = get_allowed_image_changes(model_name, original_image)
    hacked_image = np.copy(original_image)
    pre_activation = get_pre_activation(model, neuron)[0]
    prenon_gradient_function = K.gradients(K.mean(pre_activation), model_input_layer)[0]

    neuron_layer_idx = dfg_util.get_layer_index_by_node_name(neuron)
    neuron_idx = dfg_util.get_node_idx_by_node_name(neuron)
    neuron_layer_outputs = model.layers[neuron_layer_idx].output[0]
    neuron_weights= nn_util.get_layer_weights(model, neuron_layer_idx)
    b = 0
    if len(neuron_weights) == 2:
        b = neuron_weights[1][..., neuron_idx]

    ret_arr = []
    ret_arr.append(prenon_gradient_function)
    ret_arr.append(pre_activation)
    ret_arr.append(neuron_layer_outputs)
    functor = K.function([model_input_layer, K.learning_phase()], ret_arr)

    neuron_activation_flag = False
    iterations_count = 0
    while neuron_activation_flag == False:
        if (iterations_count > MAX_ITERATIONS):
            print("Update nodes failed, iterations count: {0}) > {1}".format(iterations_count, MAX_ITERATIONS))
            break
        ret_arr_vals = functor([hacked_image, 0])
        prenon_gradients = ret_arr_vals[0]
        pre_nonlineared_val = ret_arr_vals[1]
        neuron_layer_output_vals = ret_arr_vals[2]
        neuron_layer_output_val = neuron_layer_output_vals[..., neuron_idx]
        print("Updating neuron {0}, the pre-nonlineared val: {1}, bias: {2}, the output val: {3}.".format(neuron, np.mean(pre_nonlineared_val), np.mean(b), np.mean(neuron_layer_output_val)))
        # if np.all(prenon_gradients == 0):
        #     print("The greadients are all zero.")
        #     break
        if np.any(neuron_layer_output_vals[..., neuron_idx]!=0):
            neuron_activation_flag = is_neuron_activated(neuron_idx, neuron_layer_output_vals, t_func, args[0])
        if neuron_activation_flag == False:
            hacked_image += np.sign(prenon_gradients) * esp
            hacked_image = clip_image(model_name, max_change_below, max_change_above, hacked_image)
        iterations_count += 1
    hacked_prediction = np.argmax(model.predict(hacked_image)[0])
    org_prediction = np.argmax(model.predict(original_image)[0])
    updeted_image = data_util.deprocess_image(hacked_image[0], model_name)
    return neuron_activation_flag, (org_prediction, hacked_prediction), updeted_image


def get_each_layer_uncovered_edges(model_json, covered_dfg, complete_dfg, output_shapes, exclude_layers=[]):
    uncovered_edges = {}
    for layer_idx in range(1, len(output_shapes)):
        uncovered_edges[layer_idx] = []
    for layer_idx in range(len(output_shapes) - 1, -1, -1):
        if layer_idx in exclude_layers:
            continue
        pre_layer_idxes = nn_util.get_precursor_layer_idxes(model_json, layer_idx)
        for pre_layer_idx in pre_layer_idxes:
            for pre_node_idx in range(0, output_shapes[pre_layer_idx][-1]):
                pre_node_name = dfg_util.get_node_name(pre_layer_idx, pre_node_idx)
                if pre_node_name not in covered_dfg:
                    uncovered_nodes = complete_dfg[pre_node_name]
                else:
                    covered_edges = set(covered_dfg[pre_node_name])
                    complete_edges = set(complete_dfg[pre_node_name])
                    uncovered_nodes = complete_edges - covered_edges
                for uncovered_node in uncovered_nodes:
                    uncovered_layer = dfg_util.get_layer_index_by_node_name(uncovered_node)
                    uncovered_edges[uncovered_layer].append([pre_node_name, uncovered_node])
    return uncovered_edges


def get_edge_started_layers_outputs_tensors(model, edge, nodes_penalties):
    to_update_prenon_nodes = []
    layers_outputs = []
    for node_idx in range(0, len(edge)):
        node_name = edge[node_idx]
        # pre-nonlinear related info
        pre_activation = get_pre_activation(model, node_name)[0]
        pre_activation = K.mean(pre_activation)
        pre_activation = tf.minimum(pre_activation, nodes_penalties[node_idx])
        to_update_prenon_nodes.append(pre_activation)
    to_activate_prenon_nodes_tens = K.stack(to_update_prenon_nodes)

    layers_with_outputs = []
    first_layer_idx = dfg_util.get_layer_index_by_node_name(edge[0])
    for layer_idx in range(first_layer_idx, len(model.layers)):
        layers_outputs.append(model.layers[layer_idx].output[0])
        layers_with_outputs.append(layer_idx)
    return to_activate_prenon_nodes_tens, layers_with_outputs, layers_outputs


# compute controbutions from the second layer of the edge to the final layer of the DNN
def get_edge_started_contribution_tensors(model, edge, contri_and_scdn_penalties=[]):
    model_json = json.loads(model.to_json())
    fst_layer_idx = dfg_util.get_layer_index_by_node_name(edge[0])
    exclude_layers = [i for i in range(0, fst_layer_idx + 1)]
    from nn_contribution_util import build_contributions_computation_graph
    to_be_computed_layers, edge_started_layers_contris = build_contributions_computation_graph(model, exclude_layers)
    edge_started_layers_contris_over_one_img = []
    for contri in edge_started_layers_contris:
        edge_started_layers_contris_over_one_img.append(contri[0])

    contri_from_fst_layer = edge_started_layers_contris_over_one_img[0]
    to_update_contri_and_node = []
    fst_node_name = edge[0]
    fst_node_idx = dfg_util.get_node_idx_by_node_name(fst_node_name)
    scd_node_name = edge[1]
    scd_layer_idx = dfg_util.get_layer_index_by_node_name(scd_node_name)
    scd_node_idx = dfg_util.get_node_idx_by_node_name(scd_node_name)
    scd_class_name = model_json["config"]["layers"][scd_layer_idx]["class_name"]
    scd_weights = model.layers[scd_layer_idx].get_weights()
    if scd_class_name in CONV_LAYERS:
        contributions_to_scdnode = contri_from_fst_layer[..., scd_node_idx]
    elif scd_class_name in DENSE_LAYERS:
        w_between_fstnode_and_scdnode = scd_weights[0][..., scd_node_idx][fst_node_idx]
        if w_between_fstnode_and_scdnode < 0:
            print("The weight that the first node connects to is positive:", w_between_fstnode_and_scdnode)
        contributions_to_scdnode = contri_from_fst_layer[..., scd_node_idx]
    contribution_from_fstnode_2_scdnode = K.mean(contributions_to_scdnode[..., fst_node_idx])
    if len(contri_and_scdn_penalties) > 0:
        contribution_from_fstnode_2_scdnode = tf.minimum(contribution_from_fstnode_2_scdnode,
                                                         contri_and_scdn_penalties[0])
    to_update_contri_and_node.append(contribution_from_fstnode_2_scdnode)

    scd_layers_output = model.layers[scd_layer_idx].output[0]
    scd_node_output = K.mean(scd_layers_output[..., scd_node_idx])
    if len(contri_and_scdn_penalties) > 0:
        scd_node_output = tf.minimum(scd_node_output, contri_and_scdn_penalties[1])
    to_update_contri_and_node.append(scd_node_output)
    to_update_contri_and_node_tens = K.stack(to_update_contri_and_node)

    return to_update_contri_and_node_tens, to_be_computed_layers, edge_started_layers_contris_over_one_img


def putback_layers_outputs_vals(model, layers_with_outputs, layers_outputs_vals):
    dnn_layers_num = len(model.layers)
    all_layers_outputs_vals = []
    for layer_idx in range(0, dnn_layers_num):
        if layer_idx in layers_with_outputs:
            temp_idx = layers_with_outputs.index(layer_idx)
            all_layers_outputs_vals.append(layers_outputs_vals[temp_idx])
        else:
            all_layers_outputs_vals.append([])
    return all_layers_outputs_vals


def putback_layers_contributions_vals(model, layers_with_contributions, edge_started_layers_contributions):
    dnn_layers_num = len(model.layers)
    all_layers_contributions_vals = []
    for layer_idx in range(0, dnn_layers_num):
        if layer_idx in layers_with_contributions:
            temp_idx = layers_with_contributions.index(layer_idx)
            all_layers_contributions_vals.append(edge_started_layers_contributions[temp_idx])
        else:
            all_layers_contributions_vals.append([])
    return all_layers_contributions_vals


def is_contribution_activated(fst_node_idx, contributions_to_scd_node, t_func=dfg_util.gt_scaled_t, *args):
    contribution_means_to_scd_node = []
    if contributions_to_scd_node.ndim > 1:
        for pre_node_idx in range(0, contributions_to_scd_node.shape[-1]):
            contribution_means_to_scd_node.append(np.mean(contributions_to_scd_node[..., pre_node_idx]))
    else:
        contribution_means_to_scd_node = contributions_to_scd_node
    contri_flags = t_func(contribution_means_to_scd_node, args[0])
    return contri_flags[fst_node_idx]


# 1.make the first and second node's output ≠0 (in this way, we can use their gradients to activate this edge)
# 2.make the two node to be activated base on the contribution approach
# def update_connected_nodes2(model, original_image, edge, model_name, esp=0.05, MAX_ITERATIONS=200,
#                             t_func=DFGUtils.gt_scaled_t, *args):
#     model_input_layer = model.layers[0].input
#     max_change_below, max_change_above = get_allowed_image_changes(model_name, original_image)
#     hacked_image = np.copy(original_image)
#     model_json = json.loads(model.to_json())
#
#     fst_node_name = edge[0]
#     fst_layer_idx = DFGUtils.get_layer_index_by_node_name(fst_node_name)
#     fst_node_idx = DFGUtils.get_node_idx_by_node_name(fst_node_name)
#     scd_node_name = edge[1]
#     scd_layer_idx = DFGUtils.get_layer_index_by_node_name(scd_node_name)
#     scd_node_idx = DFGUtils.get_node_idx_by_node_name(scd_node_name)
#
#     st = time.time()
#     # The pre-nonlearities tensors of the two node of this edge (to make the outputs of this edge be non-zero)
#     nodes_penalties = K.placeholder(len(edge))
#     to_activate_prenon_nodes_tens, layers_with_outputs, layers_outputs = get_edge_started_layers_outputs_tensors(model,
#                                                                                                                  edge,
#                                                                                                                  nodes_penalties)
#     prenon_hinge_loss = tf.reduce_sum(to_activate_prenon_nodes_tens)
#     prenon_gradients = K.gradients(prenon_hinge_loss, model_input_layer)[0]
#
#     # The tensors of the contribution from first node, the second node and second-node started layers' contributions
#     contri_and_scdn_penalties = K.placeholder(len(edge))
#     to_update_contri_and_node_tens, layers_with_contris, edge_started_layers_contris = get_edge_started_contribution_tensors(
#         model, edge, contri_and_scdn_penalties)
#     contri_and_scdnode_loss = tf.reduce_sum(to_update_contri_and_node_tens)
#     contri_and_scdnode_gradients = K.gradients(contri_and_scdnode_loss, model_input_layer)[0]
#
#     ret_arr = []
#     ret_arr.append(prenon_gradients)
#     ret_arr.append(contri_and_scdnode_gradients)
#     ret_arr.append(to_activate_prenon_nodes_tens)
#     ret_arr.append(to_update_contri_and_node_tens)
#     ret_arr.extend(layers_outputs)
#     ret_arr.extend(edge_started_layers_contris)
#     functor = K.function([model_input_layer, nodes_penalties, contri_and_scdn_penalties, K.learning_phase()], ret_arr)
#     et = time.time()
#     print("Construct computation graph take {0} s.".format(et - st))
#
#     iterations_count = 0
#     contri_activation_flag = False
#     scd_node_activation_flag = False
#
#     output_layer_idx = len(model.layers) - 1
#     nodes_penalties_vals = np.array([999.0, 999.0])
#     contri_and_scdn_penalties = np.array([999.0, 999.0])
#     sub_covered_dfg = {}
#
#     while (not contri_activation_flag) or (not scd_node_activation_flag):
#         if (iterations_count > MAX_ITERATIONS):
#             print("Update nodes failed, iterations count: {0}) > {1}".format(iterations_count, MAX_ITERATIONS))
#             break
#         ret_arr_vals = functor([hacked_image, nodes_penalties_vals, contri_and_scdn_penalties, 0])
#         prenon_gradients_vals = ret_arr_vals[0]
#         contri_and_scdnode_gradients_vals = ret_arr_vals[1]
#         to_activate_prenon_nodes_vals = ret_arr_vals[2]
#         to_update_contri_and_node_vals = ret_arr_vals[3]
#         edge_started_layers_outputs_vals = ret_arr_vals[4:4 + len(layers_outputs)]
#         edge_started_layers_contris_vals = ret_arr_vals[4 + len(layers_outputs):]
#
#         fst_layer_output_vals = edge_started_layers_outputs_vals[0]
#         scd_layer_output_vals = edge_started_layers_outputs_vals[1]
#         fst_node_output = np.mean(fst_layer_output_vals[..., fst_node_idx])
#         scd_node_output = np.mean(scd_layer_output_vals[..., scd_node_idx])
#         edge_output = [fst_node_output, scd_node_output]
#         print("Updating edge: {0}, the prenon-vals: {1}, the outputs: {2}, and the contri-scdn-vals:{3}."
#               .format(edge, to_activate_prenon_nodes_vals, edge_output, to_update_contri_and_node_vals))
#         all_layers_outputs_vals = putback_layers_outputs_vals(model, layers_with_outputs,
#                                                               edge_started_layers_outputs_vals)
#         all_layers_contris_vals = putback_layers_contributions_vals(model, layers_with_contris,
#                                                                     edge_started_layers_contris_vals)
#         # 1.Make the output of both nodes of this edge non-zero
#         condition_with_output_layer = (scd_layer_idx == output_layer_idx) and (
#                 fst_node_output == 0.0 or not scd_node_activation_flag)
#         condition_without_output_layer = (scd_layer_idx != output_layer_idx) and (
#                 fst_node_output == 0.0 or scd_node_output == 0.0)
#         if condition_with_output_layer or condition_without_output_layer:
#             gradients = prenon_gradients_vals
#             if fst_node_output != 0:
#                 if nodes_penalties_vals[0] == 999.0:
#                     nodes_penalties_vals[0] = max(fst_node_output, 0.2)
#             if scd_node_output != 0:
#                 if scd_layer_idx == output_layer_idx:
#                     if nodes_penalties_vals[1] == 999.0:
#                         nodes_penalties_vals[1] = 1.0
#                     if scd_node_output == np.max(scd_layer_output_vals):
#                         scd_node_activation_flag = True
#                 elif nodes_penalties_vals[1] == 999.0:
#                     nodes_penalties_vals[1] = max(scd_node_output, 0.2)
#         # 2.Activate the edge (i.e., the contribution from the first node to the second node)
#         else:
#             gradients = contri_and_scdnode_gradients_vals
#             contributions_to_scd_node = all_layers_contris_vals[scd_layer_idx][..., scd_node_idx]
#             contri_activation_flag = is_contribution_activated(fst_node_idx, contributions_to_scd_node, t_func, *args)
#             if scd_layer_idx == output_layer_idx:
#                 if scd_node_output == np.max(scd_layer_output_vals):
#                     scd_node_activation_flag = True
#                     if contri_and_scdn_penalties[1] == 999.0:
#                         contri_and_scdn_penalties[1] = 1.0
#                     # during the update, the first node's output maybe optimizated to zero due to the negative contribution
#                     if fst_node_output == 0:
#                         if nodes_penalties_vals[1] == 999.0:
#                             nodes_penalties_vals[1] = max(scd_node_output, 0.2)
#                         gradients = prenon_gradients_vals
#                 else:
#                     scd_node_activation_flag = False
#             else:
#                 activation_flags = DFGUtils.extract_node_activation_flags_over_one_img(model_json, fst_layer_idx,
#                                                                                        all_layers_outputs_vals,
#                                                                                        all_layers_contris_vals, t_func,
#                                                                                        *args)
#                 scd_node_activation_flag = activation_flags[scd_layer_idx][scd_node_idx]
#                 if scd_node_activation_flag:
#                     if contri_and_scdn_penalties[1] == 999.0:
#                         contri_and_scdn_penalties[1] = max(scd_node_output, 0.2)
#                     # during the update, the first node's output maybe optimizated to zero due to the negative contribution
#                     if fst_node_output == 0:
#                         if nodes_penalties_vals[1] == 999.0:
#                             nodes_penalties_vals[1] = max(scd_node_output, 0.2)
#                         gradients = prenon_gradients_vals
#
#             if contri_activation_flag:
#                 # penalty the contribution from the first node to the second node
#                 if contri_and_scdn_penalties[0] == 999.0:
#                     contri_and_scdn_penalties[0] = max(to_update_contri_and_node_vals[0], 0.2)
#                 # during the update, the scd node's output maybe optimizated to zero due to the negative contribution
#                 if scd_node_output == 0:
#                     if nodes_penalties_vals[0] == 999.0:
#                         nodes_penalties_vals[0] = max(fst_node_output, 0.2)
#                     gradients = prenon_gradients_vals
#
#         # update gradient by i-FGSM algorithm
#         if not contri_activation_flag or not scd_node_activation_flag:
#             hacked_image += np.sign(gradients) * esp
#             hacked_image = clip_image(model_name, max_change_below, max_change_above, hacked_image)
#         iterations_count += 1
#
#     hacked_prediction = np.argmax(model.predict(hacked_image)[0])
#     org_prediction = np.argmax(model.predict(original_image)[0])
#     updeted_image = DataSetUtils.deprocess_image(hacked_image[0], model_name)
#     return contri_activation_flag, (org_prediction, hacked_prediction), updeted_image, sub_covered_dfg


def get_edge_layers_outputs_tensors(model, edge, nodes_penalties):
    to_update_prenon_nodes = []
    layers_outputs = []
    for node_idx in range(0, len(edge)):
        node_name = edge[node_idx]
        layer_idx = dfg_util.get_layer_index_by_node_name(node_name)
        layers_outputs.append(model.layers[layer_idx].output[0])
        # pre-nonlinear related info
        pre_activation = get_pre_activation(model, node_name)[0]
        pre_activation = K.mean(pre_activation)
        pre_activation = tf.minimum(pre_activation, nodes_penalties[node_idx])
        to_update_prenon_nodes.append(pre_activation)
    to_activate_prenon_nodes_tens = K.stack(to_update_prenon_nodes)

    return to_activate_prenon_nodes_tens, layers_outputs


# compute controbutions from the second layer of the edge to the final layer of the DNN
def get_contri_and_scdlayer_tensors(model, edge, contri_and_scdn_penalties):
    model_json = json.loads(model.to_json())
    to_update_contri_and_node = []
    fst_node_name = edge[0]
    fst_node_idx = dfg_util.get_node_idx_by_node_name(fst_node_name)
    scd_node_name = edge[1]
    scd_layer_idx = dfg_util.get_layer_index_by_node_name(scd_node_name)
    scd_node_idx = dfg_util.get_node_idx_by_node_name(scd_node_name)
    scd_class_name = model_json["config"]["layers"][scd_layer_idx]["class_name"]
    scd_weights = model.layers[scd_layer_idx].get_weights()
    scd_layer_input = model.layers[scd_layer_idx].input
    scd_layers_output = model.layers[scd_layer_idx].output[0]
    if scd_class_name in CONV_LAYERS:
        strides = tuple(model_json["config"]["layers"][scd_layer_idx]["config"]["strides"])
        padding = model_json["config"]["layers"][scd_layer_idx]["config"]["padding"]
        data_format = model_json["config"]["layers"][scd_layer_idx]["config"]["data_format"]
        dilation_rate = tuple(model_json["config"]["layers"][scd_layer_idx]["config"]["dilation_rate"])
        kernel = scd_weights[0]
        kernel_of_scd_node = K.expand_dims(kernel[..., scd_node_idx])
        in_contributions = K.depthwise_conv2d(scd_layer_input, kernel_of_scd_node, strides, padding, data_format,
                                              dilation_rate)
        in_contributions = in_contributions[0]
    elif scd_class_name in DENSE_LAYERS:
        w = scd_weights[0]
        w_of_scd_node = w[..., scd_node_idx]
        fst_n_w = w_of_scd_node[fst_node_idx]
        if fst_n_w < 0:
            print("The weight that the first node connects to is positive:", fst_n_w)
        in_contributions = scd_layer_input * w_of_scd_node
        in_contributions = in_contributions[0]
    in_contribution = K.mean(in_contributions[..., fst_node_idx])
    in_contribution = tf.minimum(in_contribution, contri_and_scdn_penalties[0])
    to_update_contri_and_node.append(in_contribution)

    scd_node_output = K.mean(scd_layers_output[..., scd_node_idx])
    scd_node_output = tf.minimum(scd_node_output, contri_and_scdn_penalties[1])
    to_update_contri_and_node.append(scd_node_output)
    to_update_contri_and_node_tens = K.stack(to_update_contri_and_node)
    return to_update_contri_and_node_tens, in_contributions, scd_layers_output


def caculate_node_activation_state(model, original_image, node_name, t_func=dfg_util.gt_scaled_t, *args):
    model_json = json.loads(model.to_json())
    cur_layer_idx = dfg_util.get_layer_index_by_node_name(node_name)
    cur_node_idx = dfg_util.get_node_idx_by_node_name(node_name)
    layers_with_outputs = []
    layers_outputs = []
    pre_layer_idx = nn_util.get_precursor_layer_idxes(model_json, cur_layer_idx)[0]
    for tmp_layer_idx in range(pre_layer_idx, len(model.layers)):
        layers_outputs.append(model.layers[tmp_layer_idx].output[0])
        layers_with_outputs.append(tmp_layer_idx)

    exclude_layers = [i for i in range(0, cur_layer_idx)]
    from nn_contribution_util import build_contributions_computation_graph
    to_be_computed_layers, contris = build_contributions_computation_graph(model, exclude_layers)
    layers_contris = []
    for contri in contris:
        layers_contris.append(contri[0])

    model_input_layer = model.layers[0].input
    ret_arr = []
    ret_arr.extend(layers_outputs)
    ret_arr.extend(layers_contris)
    functor = K.function([model_input_layer, K.learning_phase()], ret_arr)
    ret_arr_vals = functor([original_image, 0])
    layers_outputs_vals = ret_arr_vals[0:len(layers_with_outputs)]
    layers_contris_vals = ret_arr_vals[len(layers_with_outputs):]

    all_layers_outputs_vals = putback_layers_outputs_vals(model, layers_with_outputs, layers_outputs_vals)
    all_layers_contris_vals = putback_layers_contributions_vals(model, to_be_computed_layers, layers_contris_vals)
    activation_flags = dfg_util.extract_node_activation_flags_over_one_img(model_json, pre_layer_idx,
                                                                           all_layers_outputs_vals,
                                                                           all_layers_contris_vals, t_func, *args)
    node_activation_flag = activation_flags[cur_layer_idx][cur_node_idx]
    return node_activation_flag


# 1.make the first and second node's output ≠0 (in this way, we can use their gradients to activate this edge)
# 2.make the two node to be activated base on the contribution approach
def update_connected_nodes3(model, original_image, edge, model_name, esp=0.05, MAX_ITERATIONS=200,
                            t_func=dfg_util.gt_scaled_t, *args):
    check_step = 10
    model_input_layer = model.layers[0].input
    max_change_below, max_change_above = get_allowed_image_changes(model_name, original_image)
    hacked_image = np.copy(original_image)

    fst_node_name = edge[0]
    fst_node_idx = dfg_util.get_node_idx_by_node_name(fst_node_name)
    scd_node_name = edge[1]
    scd_layer_idx = dfg_util.get_layer_index_by_node_name(scd_node_name)
    scd_node_idx = dfg_util.get_node_idx_by_node_name(scd_node_name)

    st = time.time()
    # The pre-nonlearities tensors of the two node of this edge (to make the outputs of this edge be non-zero)
    nodes_penalties = K.placeholder(len(edge))
    prenon_nodes_tens, layers_outputs = get_edge_layers_outputs_tensors(model, edge, nodes_penalties)
    prenon_hinge_loss = tf.reduce_sum(prenon_nodes_tens)
    prenon_gradients = K.gradients(prenon_hinge_loss, model_input_layer)[0]

    # The tensors of the contribution from first node, the second node and second-node started layers' contributions
    contri_and_scdn_penalties = K.placeholder(len(edge))
    contri_and_scdnode_tens, contributions_to_scdnode, scd_layer_outputs = get_contri_and_scdlayer_tensors(model, edge,
                                                                                                           contri_and_scdn_penalties)
    contri_and_scdnode_loss = tf.reduce_sum(contri_and_scdnode_tens)
    contri_and_scdnode_gradients = K.gradients(contri_and_scdnode_loss, model_input_layer)[0]

    ret_arr = []
    ret_arr.append(prenon_gradients)
    ret_arr.append(contri_and_scdnode_gradients)
    ret_arr.append(prenon_nodes_tens)
    ret_arr.append(contributions_to_scdnode)
    ret_arr.extend(layers_outputs)
    functor = K.function([model_input_layer, nodes_penalties, contri_and_scdn_penalties, K.learning_phase()], ret_arr)
    et = time.time()
    print("Construct computation graph take {0} s.".format(et - st))

    iterations_count = 0
    contri_activation_flag = False
    scd_neuron_activation_flag = False
    scd_node_activation_flag = False

    output_layer_idx = len(model.layers) - 1
    nodes_penalties_vals = np.array([999.0, 999.0])
    contri_and_scdn_penalties = np.array([999.0, 999.0])

    while (not contri_activation_flag) or (not scd_node_activation_flag):
        if (iterations_count > MAX_ITERATIONS):
            print("Update nodes failed, iterations count: {0}) > {1}".format(iterations_count, MAX_ITERATIONS))
            break
        ret_arr_vals = functor([hacked_image, nodes_penalties_vals, contri_and_scdn_penalties, 0])
        prenon_gradients_vals = ret_arr_vals[0]
        contri_and_scdnode_gradients_vals = ret_arr_vals[1]
        prenon_nodes_vals = ret_arr_vals[2]
        contributions_to_scdnode_vals = ret_arr_vals[3]
        edge_layers_outputs_vals = ret_arr_vals[4:6]

        fst_layer_output_vals = edge_layers_outputs_vals[0]
        scd_layer_output_vals = edge_layers_outputs_vals[1]
        fst_node_output = np.mean(fst_layer_output_vals[..., fst_node_idx])
        scd_node_output = np.mean(scd_layer_output_vals[..., scd_node_idx])
        edge_output = [fst_node_output, scd_node_output]
        contri_from_fstnode_2_scdnode = np.mean(contributions_to_scdnode_vals[..., fst_node_idx])
        contri_and_scdnode_vals = [contri_from_fstnode_2_scdnode, scd_node_output]
        print("Updating edge: {0}, the prenon-vals: {1}, the outputs: {2}, and the contri_and_scdnode_vals:{3}.".format(
            edge, prenon_nodes_vals, edge_output, contri_and_scdnode_vals))
        # 1.Make the output of both nodes of this edge non-zero
        condition_with_output_layer = (scd_layer_idx == output_layer_idx) and (
                fst_node_output == 0.0 or not scd_node_activation_flag)
        condition_without_output_layer = (scd_layer_idx != output_layer_idx) and (
                fst_node_output == 0.0 or scd_node_output == 0.0)
        if condition_with_output_layer or condition_without_output_layer:
            gradients = prenon_gradients_vals
            if fst_node_output != 0:
                if nodes_penalties_vals[0] == 999.0:
                    nodes_penalties_vals[0] = max(fst_node_output, 0.2)
            if scd_node_output != 0:
                if scd_layer_idx == output_layer_idx:
                    if scd_node_output == np.max(scd_layer_output_vals):
                        scd_neuron_activation_flag = True
                        scd_node_activation_flag = True
                        if nodes_penalties_vals[1] == 999.0:
                            nodes_penalties_vals[1] = max(scd_node_output, 0.2)
                elif nodes_penalties_vals[1] == 999.0:
                    nodes_penalties_vals[1] = max(scd_node_output, 0.2)
        # 2.Activate the edge (i.e., the contribution from the first node to the second node)
        else:
            gradients = contri_and_scdnode_gradients_vals
            contri_activation_flag = is_contribution_activated(fst_node_idx, contributions_to_scdnode_vals, t_func, *args)
            if scd_layer_idx == output_layer_idx:
                if scd_node_output == np.max(scd_layer_output_vals):
                    scd_node_activation_flag = True
                    scd_neuron_activation_flag = True
                    if contri_and_scdn_penalties[1] == 999.0:
                        contri_and_scdn_penalties[1] = 1.0
                    # during the update, the first node's output maybe optimizated to zero due to the negative contribution
                    if fst_node_output == 0:
                        if nodes_penalties_vals[1] == 999.0:
                            nodes_penalties_vals[1] = max(scd_node_output, 0.2)
                        gradients = prenon_gradients_vals
                else:
                    scd_node_activation_flag = False
                    scd_neuron_activation_flag = False
            else:
                # use the neuron activation state to pre_judge the activation of the second node
                scd_neuron_activation_flag = is_neuron_activated(scd_node_idx, scd_layer_output_vals, t_func, *args)
                if scd_neuron_activation_flag:
                    # if scd_node_activation_flag:
                    if contri_and_scdn_penalties[1] == 999.0:
                        contri_and_scdn_penalties[1] = max(scd_node_output, 0.2)
                    # during the update, the first node's output maybe optimizated to zero due to the negative contribution
                    if fst_node_output == 0:
                        if nodes_penalties_vals[1] == 999.0:
                            nodes_penalties_vals[1] = max(scd_node_output, 0.2)
                        gradients = prenon_gradients_vals
            if contri_activation_flag:
                # penalty the contribution from the first node to the second node
                if contri_and_scdn_penalties[0] == 999.0:
                    contri_and_scdn_penalties[0] = max(contri_from_fstnode_2_scdnode, 0.2)
                # during the update, the scd node's output maybe optimizated to zero due to the negative contribution
                if scd_node_output == 0:
                    if nodes_penalties_vals[0] == 999.0:
                        nodes_penalties_vals[0] = max(fst_node_output, 0.2)
                    gradients = prenon_gradients_vals
                if scd_neuron_activation_flag:
                    scd_node_activation_flag = scd_neuron_activation_flag
                    # if iterations_count % check_step == 0:
                    #     scd_node_activation_flag = caculate_node_activation_state(model, hacked_image, scd_node_name, t_func, *args)
                    # scd_node_activation_flag = caculate_node_activation_state(model, hacked_image, scd_node_name, t_func, *args)
                    # print("When scd_neuron_activation_flag = True, the scd_node_activation_flag = {0}.".format(scd_node_activation_flag))
                    if scd_node_activation_flag == False:
                        nodes_penalties_vals[1] = 999.0
                        contri_and_scdn_penalties[1] = 999.0
        # update gradient by i-FGSM algorithm
        if not contri_activation_flag or not scd_node_activation_flag:
            hacked_image += np.sign(gradients) * esp
            hacked_image = clip_image(model_name, max_change_below, max_change_above, hacked_image)
        iterations_count += 1

    hacked_prediction = np.argmax(model.predict(hacked_image)[0])
    org_prediction = np.argmax(model.predict(original_image)[0])
    updeted_image = data_util.deprocess_image(hacked_image[0], model_name)
    return contri_activation_flag, (org_prediction, hacked_prediction), updeted_image


def edge_guided_generator(model_name, uncovered_edges, generated_img_path, start_seed_id, inter_results_file, threshold):
    t_func = dfg_util.gt_scaled_t
    def sample_seed(seed_arr, sample_base, seed_idx, data_type):
        x_idx = seed_arr[seed_idx]
        seed_idx += 1
        seed_idx = seed_idx % sample_base
        if data_type == 'mnist':
            _, (x_test, _) = data_util.get_mnist_data()
            x = x_test[x_idx:x_idx + 1]
        elif data_type == 'imagenet':
            x = data_util.get_imgnet_test_data(st_idx=x_idx, end_idx=x_idx + 1)
        return seed_idx, x

    if model_name in ['lenet1', 'lenet4', 'lenet5']:
        model_path = './models/{0}-relu'.format(model_name)
        model = load_model(model_path)
        data_type = 'mnist'
        sample_base = 10000
    elif model_name in ['vgg19']:
        model = VGG19(weights='imagenet')
        data_type = 'imagenet'
        sample_base = 100000
    elif model_name in ['resnet50']:
        model = ResNet50(weights='imagenet')
        data_type = 'imagenet'
        sample_base = 100000
    else:
        print("Un-support model name: {0}".format(model_name))
    model.summary()
    random.seed(1)
    seed_arr = random.sample(range(1, sample_base + 1), sample_base)
    MAX_EDGE_UPDATE_BUGET = 5

    adversarial_img_count = {}
    failed_updated_img_count = {}
    successivly_activated_edges = {}
    failed_to_activate_edges = {}

    for edge_idx in range(0, len(uncovered_edges)):
        uncovered_edge = uncovered_edges[edge_idx]
        layer_idx = nn_util.get_layer_index_by_node_name(uncovered_edge[1])
        if layer_idx not in adversarial_img_count:
            adversarial_img_count[layer_idx] = 0
        if layer_idx not in failed_updated_img_count:
            failed_updated_img_count[layer_idx] = 0
        if layer_idx not in successivly_activated_edges:
            successivly_activated_edges[layer_idx] = []
        if layer_idx not in failed_to_activate_edges:
            failed_to_activate_edges[layer_idx] = []

        edge_activation_flag = False
        edge_update_count = 0
        while edge_update_count < MAX_EDGE_UPDATE_BUGET:
            print("The {0}-th time to activate the {1}-th edge {2} of total {3} edge.".format(edge_update_count, edge_idx, uncovered_edge, len(uncovered_edges)))
            start_seed_id, x = sample_seed(seed_arr, sample_base, start_seed_id, data_type)
            ret = update_connected_nodes3(model, x, uncovered_edge, model_name, 0.05, 200, t_func, threshold)
            edge_update_count += 1
            edge_activation_flag = ret[0]
            org_pred, hacked_pred = ret[1]
            updeted_image = ret[2]

            x_idx = seed_arr[start_seed_id - 1]
            if edge_activation_flag:
                print("Successfully uses {0} to activate edge {1}.".format(x_idx, uncovered_edge))
                successivly_activated_edges[layer_idx].append(uncovered_edge)
                if org_pred != hacked_pred:
                    adversarial_img_count[layer_idx] += 1
                updated_img_file = generated_img_path + "/img_" + str(x_idx) + '_' + str(org_pred) + '_' + str(
                    hacked_pred) + '.png'
                im = Image.fromarray(updeted_image.astype(np.uint8))
                im.save(updated_img_file)
                break
            else:
                failed_updated_img_count[layer_idx] += 1
                print("Failed to use {0} to activate edge {1}.".format(x_idx, uncovered_edge))
        if edge_update_count >= MAX_EDGE_UPDATE_BUGET and edge_activation_flag == False:
            print("After {0} attempts, failed to activate edge {0}".format(edge_update_count, uncovered_edge))
            failed_to_activate_edges[layer_idx].append(uncovered_edge)

    inter_results = {}
    inter_results['start_seed_id'] = start_seed_id
    inter_results['successivly_activated_edges'] = successivly_activated_edges
    inter_results['failed_to_activate_edges'] = failed_to_activate_edges
    inter_results['adversarial_img_count'] = adversarial_img_count
    inter_results['failed_updated_img_count'] = failed_updated_img_count
    write_pkl_to_file(inter_results, inter_results_file, 'wb')

    K.clear_session()
    del model
    return start_seed_id, (successivly_activated_edges, failed_to_activate_edges, adversarial_img_count, failed_updated_img_count)


def edge_guided_generator_main():
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

    model_json = json.loads(model.to_json())
    no_weight_layers = nn_util.get_exclude_layers(model_json)
    threshold = 0

    t_path = "gt_" + str(threshold)
    dfg_base_dir = "./outputs/" + data_type + "/" + model_name + "/dfgs/" + t_path + "/"
    covered_dfg_file = dfg_base_dir + "total_covered_dfg.pkl"
    covered_dfg = read_pkl_from_file(covered_dfg_file)
    complete_dfg_file = "./outputs/" + data_type + "/" + model_name + "/dfgs/complete_dfg/complete_dfg.pkl"
    if not os.path.exists(complete_dfg_file):
        complete_dfg = dfg_util.construct_complete_dfg(model)
        write_pkl_to_file(complete_dfg, complete_dfg_file, 'wb')
    else:
        complete_dfg = read_pkl_from_file(complete_dfg_file)
    output_shapes = nn_util.get_dnn_each_layer_output_shape(model)
    uncovered_edges_of_each_layer = get_each_layer_uncovered_edges(model_json, covered_dfg, complete_dfg, output_shapes, no_weight_layers)

    to_be_activated_edges = copy.deepcopy(uncovered_edges_of_each_layer)
    cannt_activated_edges={}
    if model_name == 'vgg19':
        exclude_nodes = ['24_391', '24_1269', '24_1345', '24_2184', '24_2287', '24_2310', '24_2393', '24_2801', '24_2823', '24_3030', '24_3332', '24_3883', '24_4059']
    elif model_name == 'resnet50':
        exclude_nodes = ['53_23','31_22','18_31', '18_72', '18_210', '18_250']
    if model_name in ['vgg19','resnet50']:
        for layer_idx in to_be_activated_edges:
            for edge_idx in range(0, len(uncovered_edges_of_each_layer[layer_idx])):
                uncovered_edge = uncovered_edges_of_each_layer[layer_idx][edge_idx]
                fst_node = uncovered_edge[0]
                scd_node = uncovered_edge[1]
                if fst_node in exclude_nodes or scd_node in exclude_nodes:
                    if layer_idx not in cannt_activated_edges:
                        cannt_activated_edges[layer_idx]=[]
                    cannt_activated_edges[layer_idx].append(uncovered_edge)
                    to_be_activated_edges[layer_idx].remove(uncovered_edge)

    generated_img_path = dfg_base_dir + "/generated_imgs/"
    if os.path.exists(generated_img_path):
        shutil.rmtree(generated_img_path)
    os.mkdir(generated_img_path)

    successivly_activated_edges = {}
    failed_to_activate_edges = {}
    adversarial_img_count = {}
    failed_updated_img_count = {}

    start_seed_id = 0
    inter_results_file = dfg_base_dir + "/generator_inter_results.pkl"
    each_layer_edge_budget = 1000
    processed_uncovered_edges_info={}
    dnn_layer_num = len(model_json['config']['layers'])
    for layer_idx in range(dnn_layer_num - 1, 0, -1):
        class_name = model_json["config"]["layers"][layer_idx]["class_name"]
        if class_name in FLATTEN_LAYERS:
            continue
        if len(to_be_activated_edges[layer_idx]) == 0:
            continue
        processed_uncovered_edges_info[layer_idx] = {}
        processed_uncovered_edges_info[layer_idx]['org_uncovered_edges_num'] = len(to_be_activated_edges[layer_idx])
        cur_layer_uncovered_edges = to_be_activated_edges[layer_idx]
        if len(cur_layer_uncovered_edges)>each_layer_edge_budget:
            sampled_edges_of_this_layer=[]
            random.seed(1)
            cur_layer_edge_seed_arr = random.sample(range(1, each_layer_edge_budget + 1), each_layer_edge_budget)
            for seed in cur_layer_edge_seed_arr:
                sampled_edges_of_this_layer.append(cur_layer_uncovered_edges[seed])
        else:
            sampled_edges_of_this_layer = cur_layer_uncovered_edges
        sampled_edges_of_this_layer = sampled_edges_of_this_layer[0:100]

        processed_uncovered_edges_info[layer_idx]['sampled_uncovered_edges_num'] = len(sampled_edges_of_this_layer)
        print("Start to activate {0}-th layer's {1} edges".format(layer_idx, len(sampled_edges_of_this_layer)))
        edge_guided_generator(model_name, sampled_edges_of_this_layer, generated_img_path, start_seed_id, inter_results_file, threshold)
        inter_results = read_pkl_from_file(inter_results_file)
        start_seed_id = inter_results['start_seed_id']
        batch_successivly_activated_edges = inter_results['successivly_activated_edges']
        batch_failed_to_activate_edges = inter_results['failed_to_activate_edges']
        batch_adversarial_img_count = inter_results['adversarial_img_count']
        batch_failed_updated_img_count = inter_results['failed_updated_img_count']

        for layer_idx in batch_successivly_activated_edges:
            if layer_idx not in successivly_activated_edges:
                successivly_activated_edges[layer_idx] = batch_successivly_activated_edges[layer_idx]
            else:
                successivly_activated_edges[layer_idx] += batch_successivly_activated_edges[layer_idx]
        for layer_idx in batch_failed_to_activate_edges:
            if layer_idx not in failed_to_activate_edges:
                failed_to_activate_edges[layer_idx] = batch_failed_to_activate_edges[layer_idx]
            else:
                failed_to_activate_edges[layer_idx] += batch_failed_to_activate_edges[layer_idx]
        for layer_idx in batch_adversarial_img_count:
            if layer_idx not in adversarial_img_count:
                adversarial_img_count[layer_idx] = batch_adversarial_img_count[layer_idx]
            else:
                adversarial_img_count[layer_idx] += batch_adversarial_img_count[layer_idx]
        for layer_idx in batch_failed_updated_img_count:
            if layer_idx not in failed_updated_img_count:
                failed_updated_img_count[layer_idx] = batch_failed_updated_img_count[layer_idx]
            else:
                failed_updated_img_count[layer_idx] += batch_failed_updated_img_count[layer_idx]

    each_layer_successivly_activated_edge_count = {}
    total_successivly_activated_edge_count = 0
    for layer_idx in successivly_activated_edges:
        each_layer_successivly_activated_edge_count[layer_idx] = len(successivly_activated_edges[layer_idx])
        total_successivly_activated_edge_count += each_layer_successivly_activated_edge_count[layer_idx]

    each_layer_failed_to_activate_edge_count = {}
    total_failed_to_activate_edge_count = 0
    for layer_idx in successivly_activated_edges:
        each_layer_failed_to_activate_edge_count[layer_idx] = len(failed_to_activate_edges[layer_idx])
        total_failed_to_activate_edge_count += each_layer_failed_to_activate_edge_count[layer_idx]

    each_layer_adversarial_img_count = adversarial_img_count
    total_adversarial_img_count = 0
    for layer_idx in each_layer_adversarial_img_count:
        total_adversarial_img_count += each_layer_adversarial_img_count[layer_idx]

    each_layer_failed_img_count = failed_updated_img_count
    total_failed_updated_img_count = 0
    for layer_idx in each_layer_failed_img_count:
        total_failed_updated_img_count += each_layer_failed_img_count[layer_idx]

    generator_results = {}
    generator_results['start_seed_id'] = start_seed_id
    generator_results['successivly_activated_edges'] = successivly_activated_edges
    generator_results['failed_to_activate_edges'] = failed_to_activate_edges
    generator_results['adversarial_img_count'] = adversarial_img_count
    generator_results['failed_updated_img_count'] = failed_updated_img_count
    results_file = dfg_base_dir + "/generator_results.plk"
    write_pkl_to_file(generator_results, results_file, 'wb')
    print('each_layer_successivly_activated_edge_count :', each_layer_successivly_activated_edge_count,
          ', total_successivly_activated_edge_count :', total_successivly_activated_edge_count,
          ', each_layer_failed_to_activate_edge_count :', each_layer_failed_to_activate_edge_count,
          ', total_failed_to_activate_edge_count:', total_failed_to_activate_edge_count,
          ', each_layer_adversarial_img_count:', each_layer_adversarial_img_count,
          ', total_adversarial_img_count:', total_adversarial_img_count,
          ', each_layer_failed_img_count:', each_layer_failed_img_count,
          ', total_failed_updated_img_count:', total_failed_updated_img_count,
          'processed_uncovered_edges_info', processed_uncovered_edges_info)


def get_uncovered_neurons(neuron_states):
    uncovered_neurons = {}
    for layer_idx in range(0, len(neuron_states)):
        for node_idx in range(0, len(neuron_states[layer_idx])):
            if neuron_states[layer_idx][node_idx] == 0:
                uncovered_neuron = dfg_util.get_node_name(layer_idx, node_idx)
                if layer_idx not in uncovered_neurons:
                    uncovered_neurons[layer_idx] = []
                uncovered_neurons[layer_idx].append(uncovered_neuron)
    return uncovered_neurons


def neuron_guided_generator_mian():
    # model_name = 'lenet1'
    # model_name = 'lenet4'
    # model_name = 'lenet5'
    # model_path = './models/{0}-relu'.format(model_name)
    # model = load_model(model_path)
    # data_type = 'mnist'
    # sample_base = 10000
    # end_idx = 10000

    model = ResNet50(weights='imagenet')
    model_name = 'resnet50'
    # model = VGG19(weights='imagenet')
    # model_name = 'vgg19'
    data_type = 'imagenet'
    sample_base = 100000
    end_idx = 5000

    model_json = json.loads(model.to_json())
    node_related_exclude_layers = nn_util.get_node_related_exclude_layers(model_json)
    if data_type == 'mnist':
        _, (x_test, _) = data_util.get_mnist_data()
    random.seed(1)
    seed_arr = random.sample(range(1, sample_base + 1), sample_base)

    threshold = 0
    t_func = dfg_util.gt_scaled_t
    t_path = "gt_" + str(threshold)
    nc_base_dir = "./outputs/" + data_type + "/" + model_name + "/ncs/" + t_path + "/"
    neuron_states_file = nc_base_dir + '/all_covered_neuron_states.pkl'
    neuron_states = read_pkl_from_file(neuron_states_file)[end_idx]
    uncovered_neurons = get_uncovered_neurons(neuron_states)
    # uncovered_neurons_num=0
    # for layer_idx in uncovered_neurons:
    #     uncovered_neurons_num += len(uncovered_neurons[layer_idx])
    # print(uncovered_neurons_num)

    generated_img_path = nc_base_dir + "/generated_imgs/"
    if os.path.exists(generated_img_path):
        shutil.rmtree(generated_img_path)
    os.mkdir(generated_img_path)

    new_covered_neurons = {}
    for layer_idx in range(1, len(model.layers)):
        new_covered_neurons[layer_idx] = set()

    successivly_updated_img_count = {}
    adversarial_img_count = {}
    failed_updated_img_count = {}

    successivly_activated_neuron_count = {}
    failed_to_activate_neurons = {}

    MAX_EDGE_UPDATE_BUGET = 5
    seed_idx = 0

    def sample_seed(seed_arr, sample_base, seed_idx, data_type):
        x_idx = seed_arr[seed_idx]
        seed_idx += 1
        seed_idx = seed_idx % sample_base
        if data_type == 'mnist':
            x = x_test[x_idx:x_idx + 1]
        elif data_type == 'imagenet':
            x = data_util.get_imgnet_test_data(st_idx=x_idx, end_idx=x_idx + 1)
        return seed_idx, x

    for layer_idx in range(len(model.layers) - 1, 0, -1):
        if layer_idx in node_related_exclude_layers:
            continue
        for neuron_idx in range(0, len(neuron_states[layer_idx])):
            if neuron_states[layer_idx][neuron_idx] != 0:
                continue
            uncovered_neuron = data_util.get_node_name(layer_idx, neuron_idx)
            neuron_update_count = 0
            while neuron_update_count < MAX_EDGE_UPDATE_BUGET:
                seed_idx, x = sample_seed(seed_arr, sample_base, seed_idx, data_type)
                ret = activate_neuron(model, x, uncovered_neuron, model_name, 0.05, 1000, t_func, 0)
                neuron_update_count += 1
                neuron_activation_flag = ret[0]
                org_pred, hacked_pred = ret[1]
                updeted_image = ret[2]

                x_idx = seed_arr[seed_idx - 1]
                if neuron_activation_flag:
                    print("Successfully uses {0} to activate neuron {1}.".format(x_idx, uncovered_neuron))
                    new_covered_neurons[layer_idx].add(uncovered_neuron)
                    if layer_idx not in successivly_updated_img_count:
                        successivly_updated_img_count[layer_idx] = 0
                    successivly_updated_img_count[layer_idx] += 1
                    if layer_idx not in successivly_activated_neuron_count:
                        successivly_activated_neuron_count[layer_idx] = 0
                    successivly_activated_neuron_count[layer_idx] += 1
                    if org_pred != hacked_pred:
                        print("I find adversarial example, org {0}, adversarial {1}".format(org_pred, hacked_pred))
                        if layer_idx not in adversarial_img_count:
                            adversarial_img_count[layer_idx] = 0
                        adversarial_img_count[layer_idx] += 1
                    updated_img_file = generated_img_path + "/img_" + str(x_idx) + '_' + str(org_pred) + '_' + str(
                        hacked_pred) + '.png'
                    im = Image.fromarray(updeted_image.astype(np.uint8))
                    im.save(updated_img_file)
                    break
                else:
                    if layer_idx not in failed_updated_img_count:
                        failed_updated_img_count[layer_idx] = 0
                    failed_updated_img_count[layer_idx] += 1
                    print("Failed to use {0} to activate neuron {1}.".format(x_idx, uncovered_neuron))
            if neuron_update_count >= MAX_EDGE_UPDATE_BUGET and neuron_activation_flag == False:
                print("After {0} attempts, failed to activate neuron {0}".format(neuron_update_count, uncovered_neuron))
                if layer_idx not in failed_to_activate_neurons:
                    failed_to_activate_neurons[layer_idx] = []
                failed_to_activate_neurons[layer_idx].append(uncovered_neuron)
    test_gen_result_file=nc_base_dir+"test_gen_result.json"
    test_gen_result = {}
    test_gen_result['successivly_updated_img_count'] = successivly_updated_img_count
    test_gen_result['adversarial_img_count'] = adversarial_img_count
    test_gen_result['successivly_activated_neuron_count'] = successivly_activated_neuron_count
    failed_activated_neuron_count = 0
    for layer_idx in failed_to_activate_neurons:
        failed_activated_neuron_count += len(failed_to_activate_neurons[layer_idx])
    test_gen_result['failed_activated_neuron_count'] = failed_activated_neuron_count
    test_gen_result['failed_to_activate_neurons'] = failed_to_activate_neurons
    test_gen_result['failed_updated_img_count'] = failed_updated_img_count
    print_err(test_gen_result)
    write_json_to_file(test_gen_result, test_gen_result_file, 'w')


if __name__ == '__main__':
    print("Hello, Generator")
    # neuron_guided_generator_mian()
    edge_guided_generator_main()
