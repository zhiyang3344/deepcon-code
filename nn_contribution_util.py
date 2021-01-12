import numpy as np
from ioutil import *
import json
import time
import multiprocessing
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGENET_MODELS = ['vgg19', 'resnet50']
MNIST_MODELS = ['lenet1', 'lenet4', 'lenet5']


def output_dnn_contributions_in_batches(model, st_idx=0, end_idx=10000, batch_size=20):
    import data_util
    model_name = model.name
    if model_name in MNIST_MODELS:
        _, (x_test, _) = data_util.get_mnist_data()
    for batch_idx in range(st_idx, end_idx, batch_size):
        batch_st_idx = batch_idx
        batch_end_idx = batch_st_idx + batch_size
        post_fix = str(batch_st_idx) + "-" + str(batch_end_idx)
        if model_name in MNIST_MODELS:
            x = x_test[batch_st_idx:batch_end_idx]
            data_type = 'mnist'
        elif model_name in IMAGENET_MODELS:
            x = data_util.get_imgnet_test_data(batch_st_idx, batch_end_idx)
            data_type = 'imagenet'

        base_dir = "./outputs/" + data_type + "/" + model_name + "/contris/"
        print(">>>======== extract contributions over inputs {0} ==========>>>.".format(post_fix))
        contris, _ = output_contributions(model, x, is_auto_clear_session=True)
        contris_file = base_dir + "/contris_" + post_fix + ".pkl"
        write_pkls_to_file(contris, contris_file, 'wb')


def compute_features_means(featuremap_num, cur_layer_out, cur_img_depthwise_conv2d_vals):
    ci = [[] for i in range(0, featuremap_num)]
    for featuremap_idx in range(0, featuremap_num):
        if np.all(cur_layer_out[..., featuremap_idx] == 0):
            continue
        # start:end:stride
        chn_contrib_vals_of_cur_img = cur_img_depthwise_conv2d_vals[:, :, featuremap_idx::featuremap_num]
        means = []
        for c_idx in range(0, chn_contrib_vals_of_cur_img.shape[-1]):
            means.append(np.mean(chn_contrib_vals_of_cur_img[..., c_idx]))
        ci[featuremap_idx] = np.array(means)
    return ci


# def get_all_layers_in_vals(model, x):
#     from keras import backend as K
#     model_input_layer = model.layers[0].input
#     all_layers_ins = []
#     for layer_idx in range(2, len(model.layers)):
#         if isinstance(model.layers[layer_idx].input, list):
#             all_layers_ins.append(K.stack(model.layers[layer_idx].input))
#         else:
#             all_layers_ins.append(model.layers[layer_idx].input)
#     functor = K.function([model_input_layer, K.learning_phase()], all_layers_ins)
#     all_layers_in_vals = functor([x, 0])
#     all_layers_in_vals.insert(0, x)
#     all_layers_in_vals.insert(0, [])
#     return all_layers_in_vals


# def get_all_layers_out_vals(model, x):
#     from keras import backend as K
#     model_input_layer = model.layers[0].input
#     all_layers_outs = []
#     for layer_idx in range(1, len(model.layers)):
#         all_layers_outs.append(model.layers[layer_idx].output)
#     functor = K.function([model_input_layer, K.learning_phase()], all_layers_outs)
#     all_layers_out_vals = functor([x, 0])
#     all_layers_out_vals.insert(0, x)
#     return all_layers_out_vals


def output_contributions(model, x, exclude_layers=[], is_auto_clear_session=False):
    from keras import backend as K
    from nn_util import auto_clear_session_and_rebuild_model, get_all_layers_out_vals
    DENSE_LAYERS = ['Dense']
    CONV_LAYERS = ['Conv2D']

    if is_auto_clear_session:
        model = auto_clear_session_and_rebuild_model(model)

    model_json = json.loads(model.to_json())
    all_contributions = [[{} for i in range(0, len(model.layers))] for j in range(0, len(x))]
    all_layers_out_vals = get_all_layers_out_vals(model, x)

    model_input_layer = model.layers[0].input
    dnn_depth = len(model.layers)

    for layer_idx in range(len(model.layers) - 1, 0, -1):
        if layer_idx in exclude_layers:
            continue
        class_name = model_json["config"]["layers"][layer_idx]["class_name"]
        layer_name = model_json["config"]["layers"][layer_idx]["name"]
        print("compute the {0}-th layer ({1}) contributions: ".format(layer_idx, layer_name))
        weights = model.layers[layer_idx].get_weights()
        input = model.layers[layer_idx].input

        if class_name in CONV_LAYERS:
            strides = tuple(model_json["config"]["layers"][layer_idx]["config"]["strides"])
            padding = model_json["config"]["layers"][layer_idx]["config"]["padding"]
            data_format = model_json["config"]["layers"][layer_idx]["config"]["data_format"]
            dilation_rate = tuple(model_json["config"]["layers"][layer_idx]["config"]["dilation_rate"])
            kernel = weights[0]
            st = time.time()
            # print("start compute depthwise_conv2d of the {0}-th layer.".format(layer_idx))
            # print("start build depthwise_conv2d computation graph of the{0}-th layer.".format(layer_idx))
            depthwise_conv2d = K.depthwise_conv2d(input, kernel, strides, padding, data_format, dilation_rate)
            functor = K.function([model_input_layer, K.learning_phase()], [depthwise_conv2d])
            depthwise_conv2d_vals = functor([x, 0])[0]
            et = time.time()
            print("compute depthwise_conv2d of the {0}-th layer take {1} s.".format(layer_idx, et - st))

            # use multiprocessing to speed up
            pool = multiprocessing.Pool()
            rets = []
            st = time.time()
            for img_idx in range(0, len(x)):
                ret = pool.apply_async(compute_features_means, (
                    kernel.shape[-1], all_layers_out_vals[layer_idx][img_idx], depthwise_conv2d_vals[img_idx]))
                rets.append(ret)
            pool.close()
            pool.join()
            et = time.time()
            print("compute featuremaps of the {0}-th layer take {1} s.".format(layer_idx, et - st))
            for img_idx in range(0, len(x)):
                all_contributions[img_idx][layer_idx]['c_i'] = np.array(rets[img_idx].get())

        elif class_name in DENSE_LAYERS:
            con_weight = weights[0]
            output_neuron_num = con_weight.shape[-1]
            tiled_input = K.tile(K.expand_dims(input, -1), [1, 1, output_neuron_num])
            contributions = tiled_input * con_weight
            functor = K.function([model_input_layer, K.learning_phase()], [contributions])
            contribution_values = functor([x, 0])[0]
            for img_idx in range(0, len(x)):
                all_contributions[img_idx][layer_idx]['c_i'] = [[] for i in range(0, output_neuron_num)]
                if layer_idx == dnn_depth - 1:
                    out_neuron_idx = np.argmax(all_layers_out_vals[layer_idx][img_idx])
                    all_contributions[img_idx][layer_idx]['c_i'][out_neuron_idx] = contribution_values[img_idx][..., out_neuron_idx]
                else:
                    for out_neuron_idx in range(0, output_neuron_num):
                        if all_layers_out_vals[layer_idx][img_idx][out_neuron_idx] == 0:
                            continue
                        all_contributions[img_idx][layer_idx]['c_i'][out_neuron_idx] = contribution_values[img_idx][..., out_neuron_idx]
    return all_contributions, all_layers_out_vals


# def get_precursor_layer_idxes(model_json, cur_layer_idx):
#     layer_infos = model_json["config"]["layers"]
#     inbound_nodes = layer_infos[cur_layer_idx]["inbound_nodes"]
#     layer_idxes = []
#     if len(inbound_nodes) > 0:
#         for i in range(0, len(inbound_nodes[0])):
#             pre_layer_nm = inbound_nodes[0][i][0]
#             pre_layer_idx = get_layer_index_by_layer_name(model_json, pre_layer_nm)
#             layer_idxes.append(pre_layer_idx)
#         return layer_idxes
#     return []


# def mean_bigger_than_zero(feature_maps):
#     return mean_bigger_than_t(feature_maps, 0)


# def mean_bigger_than_t(feature_maps, t):
#     fms_cnt = feature_maps.shape[-1]
#     ret = np.zeros(fms_cnt, dtype=int)
#     for i in range(0, fms_cnt):
#         fm_mean = np.mean(feature_maps[..., i])
#         if (fm_mean > t):
#             ret[i] = 1
#     return ret


# def any_not_equal_zero(feature_maps):
#     ret = np.zeros((feature_maps.shape[-1]), dtype=int)
#     for i in range(0, feature_maps.shape[-1]):
#         if np.any(feature_maps[..., i] != 0):
#             ret[i] = 1
#     return ret


# def any_node_bigger_than_zero(feature_maps):
#     return any_node_bigger_than_t(feature_maps, 0)
#
#
# def any_node_bigger_than_t(feature_maps, t):
#     ret = np.zeros((feature_maps.shape[-1]), dtype=int)
#     for i in range(0, feature_maps.shape[-1]):
#         if np.any(feature_maps[..., i] > t):
#             ret[i] = 1
#     return ret;


# def get_argmax(feature_maps):
#     fms_cnt = feature_maps.shape[-1]
#     ret = np.zeros(fms_cnt, dtype=int)
#     if (feature_maps.ndim == 1):
#         ret[np.argmax(feature_maps)] = 1
#         return ret
#     else:
#         raise ValueError("ERROR, feature_maps must be a 1d numpy")
#         return
#
#
# def get_argmax_K(feature_maps, k):
#     if (feature_maps.ndim == 1):
#         ret_arr = np.zeros_like(feature_maps)
#         tmp_arr = feature_maps.copy()
#         for i in range(0, k):
#             max_idx = np.argmax(tmp_arr)
#             ret_arr[max_idx] = 1
#             tmp_arr[max_idx] = 0
#         return ret_arr
#     else:
#         raise ValueError("ERROR, feature_maps must be a 1d numpy")
#         return


# def output_all_featuremap(feature_maps):
#     fms_cnt = feature_maps.shape[-1]
#     return np.ones(fms_cnt, dtype=int)


# def is_node_in_graph(node, graph):
#     for connect_node in graph:
#         if node == connect_node:
#             return True
#         else:
#             for connected_node in graph[connect_node]:
#                 if node == connected_node:
#                     return True
#     return False


# def get_exclude_layers(model_json):
#     included_layers = ["Conv2D", "Dense", "BatchNormalization"]
#     layer_infos = model_json["config"]["layers"]
#     exclude_layers = []
#     for layer_index in range(0, len(layer_infos)):
#         class_name = layer_infos[layer_index]["class_name"]
#         if class_name not in included_layers:
#             exclude_layers.append(layer_index)
#     return exclude_layers


# def geq_t_pct(pre_val, t, cur_val):
#     if pre_val * t >= cur_val:
#         return True
#     return False


# def is_contain_deactivated_featuremap(featuremaps):
#     if (featuremaps.ndim == 3 or featuremaps.ndim == 1):
#         fm_cnt = featuremaps.shape[-1]
#         deactivated_fmcnt = 0
#         for fm_idx in range(0, fm_cnt):
#             if (featuremaps.ndim == 3 and np.all((featuremaps[:, :, fm_idx]) == 0)) or (
#                     featuremaps.ndim == 1 and np.all((featuremaps[fm_idx]) == 0)):
#                 # print("a deactivated featuremap found and its index is: "+str(fm_idx))
#                 deactivated_fmcnt += 1
#         if deactivated_fmcnt > 0:
#             print("INFO, the number of deactivated featuremaps: " + str(deactivated_fmcnt))
#             return True
#         return False
#     else:
#         print("WARNING, the dim of featuremaps must be 1 or 3 !")
#         return False


# # Can only handle the mapping of 3-D featur_map to flattened 1-d array
# # ret: [fm_idx, h_idx, w_idx]
# def get_featuremap2dense_mapper(feature_maps):
#     shape = feature_maps.shape
#     ndim = feature_maps.ndim
#     if (ndim == 3):
#         mapper = []
#         for fm_idx in range(0, shape[-1]):
#             # print("----------fm------------"+str(fm_idx))
#             # print(feature_maps[:,:,fm_idx])
#             tmp_arr = np.zeros((shape[0], shape[1]), dtype=int)
#             for h_idx in range(0, shape[0]):
#                 # print("----------h_idx------------" + str(h_idx))
#                 for w_idx in range(0, shape[1]):
#                     index = h_idx * shape[1] * shape[2] + w_idx * shape[2] + fm_idx
#                     # print(str(index)+" "+str(fm_fln[index]))
#                     tmp_arr[h_idx][w_idx] = index
#             mapper.append(tmp_arr)
#         return np.array(mapper)
#     print("ERROR, I can only handle 3d-featur_map!")
#     return
#
#
# # ret: {idx_in_dense:[h_idx, w_idx, fm_idx], ...}
# def get_dense2featuremap_mapper(fmlayer_output_shape):
#     shape = fmlayer_output_shape
#     ndim = len(fmlayer_output_shape)
#     # dense2fm_idxes = np.zeros((shape[0] * shape[1] * shape[2]), dtype=int)
#     dense2fm_idxes = {}
#     if (ndim == 3):
#         for fm_idx in range(0, shape[-1]):
#             for h_idx in range(0, shape[0]):
#                 for w_idx in range(0, shape[1]):
#                     index = h_idx * shape[1] * shape[2] + w_idx * shape[2] + fm_idx
#                     # dense2fm_idxes[index] = fm_idx
#                     dense2fm_idxes[index] = [h_idx, w_idx, fm_idx]
#         return dense2fm_idxes
#     print("ERROR, I can only handle 3d-featur_map!")
#     return


# def get_layer_outputs(sqmodel, x, start_layer_index=1, end_layer_index=-1):
#     from keras import backend as K
#
#     if start_layer_index == 0:
#         start_layer_index = 1
#     if end_layer_index == -1:
#         end_layer_index = len(sqmodel.layers)
#     ret_list = [[[]] * len(sqmodel.layers) for i in range(0, len(x))]
#     # concate inputs
#     for image_index in range(0, len(x)):
#         ret_list[image_index][0] = x[image_index]
#     model_input_layer = sqmodel.layers[0].input
#     ret_arr = []
#     for layer_index in range(start_layer_index, end_layer_index):
#         ret_arr.append(sqmodel.layers[layer_index].output)
#     functor = K.function([model_input_layer, K.learning_phase()], ret_arr)
#     results = functor([x, 0])
#     for idx in range(0, len(results)):
#         # print("INFO, concate the prediction in layer: {0}/{1}".format(idx + start_layer_index, end_layer_index - 1))
#         for image_index in range(0, len(results[idx])):
#             ret_list[image_index][idx + start_layer_index] = results[idx][image_index]
#     return ret_list


# def get_layer_info_by_name(layer_infos, name):
#     for layer_info in layer_infos:
#         if (layer_info['name'] == name):
#             return layer_info
#     # print("WARN, not found: " + name)
#     return
#
#
# def get_layer_index_by_layer_name(model_json, layer_name):
#     layer_infos = model_json["config"]["layers"]
#     for layer_index in range(0, len(layer_infos)):
#         if layer_infos[layer_index]["name"] == layer_name:
#             return layer_index
#     return -1


# def rearrage_depthwise_conv2ded_contributions(depthwise_conv2ded_contributions, featuremap_num):
#     from keras import backend as K
#     conv2ded_contributions = []
#     for fm_idx in range(0, featuremap_num):
#         featuremap = depthwise_conv2ded_contributions[:, :, :, fm_idx::featuremap_num]
#         conv2ded_contributions.append(featuremap)
#     conv2ded_contributions = K.stack(conv2ded_contributions,-1)
#     return conv2ded_contributions
#
#
# def build_contributions_computation_graph(model, exclude_layers=[]):
#     from keras import backend as K
#     DENSE_LAYERS = ['Dense']
#     CONV_LAYERS = ['Conv2D']
#
#     model_json = json.loads(model.to_json())
#     all_contributions = []
#     to_be_computed_layers = []
#     for layer_idx in range(1, len(model.layers)):
#         if layer_idx in exclude_layers:
#             continue
#         class_name = model_json["config"]["layers"][layer_idx]["class_name"]
#         weights = model.layers[layer_idx].get_weights()
#         if class_name in CONV_LAYERS:
#             to_be_computed_layers.append(layer_idx)
#             strides = tuple(model_json["config"]["layers"][layer_idx]["config"]["strides"])
#             padding = model_json["config"]["layers"][layer_idx]["config"]["padding"]
#             data_format = model_json["config"]["layers"][layer_idx]["config"]["data_format"]
#             dilation_rate = tuple(model_json["config"]["layers"][layer_idx]["config"]["dilation_rate"])
#             kernel = weights[0]
#             input = model.layers[layer_idx].input
#
#             depthwise_conv2d = K.depthwise_conv2d(input, kernel, strides, padding, data_format, dilation_rate)
#             contributions = rearrage_depthwise_conv2ded_contributions(depthwise_conv2d, kernel.shape[-1])
#             all_contributions.append(contributions)
#         elif class_name in DENSE_LAYERS:
#             to_be_computed_layers.append(layer_idx)
#             con_weight = weights[0]
#             input = model.layers[layer_idx].input
#
#             tiled_input = K.tile(K.expand_dims(input, -1), [1, 1, con_weight.shape[-1]])
#             contributions = tiled_input * con_weight
#             all_contributions.append(contributions)
#     return to_be_computed_layers, all_contributions


# def caculate_contributions(model, x, exclude_layers=[]):
#     from keras import backend as K
#     model_input_layer = model.layers[0].input
#     all_contributions = build_contributions_computation_graph(model, exclude_layers)
#     functor = K.function([model_input_layer, K.learning_phase()], all_contributions)
#
#     st = time.time()
#     _, all_contribution_results = functor([x, 0])
#     et = time.time()
#     print("compute contributions takes {0} s.".format(et - st))
#     return all_contribution_results



# def split_preds_files(model_name, data_type, start_idx, end_idx):
#     import shutil
#     # model_name = 'resnet50'
#     # data_type = 'imagenet'
#     # start_idx = 5000
#     # end_idx = 10000
#     org_base_dir = "./outputs/" + data_type + "/" + model_name + "/preds/"
#     dest_base_dir = "./outputs/" + data_type + "/" + model_name + "/preds_{0}-{1}/".format(start_idx, end_idx)
#     if not os.path.exists(dest_base_dir):
#         os.makedirs(dest_base_dir)
#     preds_batch_size = 20
#     for preds_batch_idx in range(start_idx, end_idx, preds_batch_size):
#         preds_start_idx = preds_batch_idx
#         preds_end_idx = preds_batch_idx + preds_batch_size
#         preds_post_fix = str(preds_start_idx) + "-" + str(preds_end_idx)
#         org_filename = org_base_dir + "/preds_" + preds_post_fix + ".pkl"
#         print('move org_filename:', org_filename, 'into dest_base_dir:', dest_base_dir)
#         shutil.move(org_filename, dest_base_dir)



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
    print("model name: {0}, start index: {1}, end index: {2}, batch size: {3}".format(model_name, st_idx, end_idx, batch_size))
    output_dnn_contributions_in_batches(model, st_idx=st_idx, end_idx=end_idx, batch_size=batch_size)
