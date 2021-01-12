import keras
from keras.models import Model
import json
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.models import load_model
import numpy as np

DENSE_LAYERS = ['Dense']
CONV_LAYERS = ['Conv2D']
FLATTEN_LAYERS = ["Flatten"]


def is_numpyarr_contain_zero(arr):
    cnt = np.sum(arr == 0)
    if (cnt > 0):
        print("the number of zero: " + str(cnt))
        return True
    return False


def output_model_structure(model):
    for ly_idx in range(0, len(model.layers)):
        layer = model.layers[ly_idx]
        weights = layer.get_weights()
        print("_________________________________________________________________")
        print("layer index: " + str(ly_idx) + " layer.name:" + str(layer.name))
        for weight in weights:
            print("weight.shape:" + str(weight.shape))
            # print("weight:" + str(weight))
            # is_numpyarr_contain_zero(weight)
        print("input.shape:" + str(layer.input_shape))
        # print("output.shape:" + str(output.shape))
        print("output.shape:" + str(layer.output_shape))
    print("=================================================================")


def get_all_layers_out_vals(model, x):
    from keras import backend as K
    model_input_layer = model.layers[0].input
    all_layers_outs = []
    for layer_idx in range(1, len(model.layers)):
        all_layers_outs.append(model.layers[layer_idx].output)
    functor = K.function([model_input_layer, K.learning_phase()], all_layers_outs)
    all_layers_out_vals = functor([x, 0])
    all_layers_out_vals.insert(0, x)
    return all_layers_out_vals


def get_dnn_each_layer_output_shape(model):
    output_shapes = []
    for layer_idx in range(0, len(model.layers)):
        output_shapes.append(model.layers[layer_idx].output_shape)
    return output_shapes


def get_dnn_each_layer_input_shape(model):
    input_shapes = []
    for layer_idx in range(0, len(model.layers)):
        input_shapes.append(model.layers[layer_idx].input_shape)
    return input_shapes


def auto_clear_session_and_rebuild_model(model):
    model_name = model.name
    K.clear_session()
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
    return model


def get_output_values_of_each_layer(org_model, x, start_index=1, end_index=-1):
    if end_index == -1:
        end_index = len(org_model.layers)
    for layer_index in range(start_index, end_index):
        tmp_model = Model(inputs=org_model.input, outputs=org_model.layers[layer_index].output)
        features = tmp_model.predict(x)
        print("_________________________________________________________________")
        print("layer index: " + str(layer_index) + ", layer.name:" + str(org_model.layers[layer_index].name))
        print("output.shape:" + str(org_model.layers[layer_index].output.shape))
        # is_numpyarr_contain_zero(features[0])
        # for img_idx in range(0, len(features)):
        # is_contain_deactivated_featuremap(features[img_idx])
        # print(features[img_idx])
    print("=================================================================")


def get_layer_weights(model, layer_idx):
    layer = model.layers[layer_idx]
    return layer.get_weights()


def decode_mnist_predictions(preds, top=5):
    CLASS_CUSTOM = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_CUSTOM[i]) + (pred[i] * 100,) for i in top_indices]
        results.append(result)
    return results


def get_exclude_layers(model_json):
    exclude_layers = []
    for layer_idx in range(0, len(model_json["config"]["layers"])):
        class_name = model_json["config"]["layers"][layer_idx]["class_name"]
        if class_name not in CONV_LAYERS + DENSE_LAYERS:
            exclude_layers.append(layer_idx)
    return exclude_layers


def get_node_related_exclude_layers(model_json):
    node_related_exclude_layers = []
    include_layers = set()
    for layer_idx in range(0, len(model_json["config"]["layers"])):
        class_name = model_json["config"]["layers"][layer_idx]["class_name"]
        if class_name in CONV_LAYERS + DENSE_LAYERS:
            include_layers.add(layer_idx)
            precursor_layer_idxes = get_precursor_layer_idxes(model_json, layer_idx)
            include_layers |= set(precursor_layer_idxes)
    for layer_idx in range(0, len(model_json["config"]["layers"])):
        if layer_idx not in include_layers:  # or class_name in FLATTEN_LAYERS:
            node_related_exclude_layers.append(layer_idx)
    return node_related_exclude_layers


def get_node_output_val(model, x, node_name):
    from keras import backend as K
    import dfg_util
    model_input_layer = model.layers[0].input
    layer_idx = dfg_util.get_layer_index_by_node_name(node_name)
    node_idx = dfg_util.get_node_idx_by_node_name(node_name)
    node_output = model.layers[layer_idx].output[..., node_idx]
    functor = K.function([model_input_layer, K.learning_phase()], [node_output])
    node_output_val = functor([x, 0])[0]
    return node_output_val


def get_node_idx_by_node_name(node_name):
    strs = node_name.split("_")
    return int(strs[1])


def get_layer_index_by_node_name(node_name):
    strs = node_name.split("_")
    return int(strs[0])


def get_node_input_val(model, x, node_name):
    from keras import backend as K
    model_input_layer = model.layers[0].input
    layer_idx = get_layer_index_by_node_name(node_name)
    node_input = model.layers[layer_idx].input
    functor = K.function([model_input_layer, K.learning_phase()], [node_input])
    node_input_val = functor([x, 0])[0]
    return node_input_val


def get_layer_index_by_layer_name(model_json, layer_name):
    layer_infos = model_json["config"]["layers"]
    for layer_index in range(0, len(layer_infos)):
        if layer_infos[layer_index]["name"] == layer_name:
            return layer_index
    return -1


def get_precursor_layer_idxes(model_json, cur_layer_idx):
    layer_infos = model_json["config"]["layers"]
    inbound_nodes = layer_infos[cur_layer_idx]["inbound_nodes"]
    layer_idxes = []
    if len(inbound_nodes) > 0:
        for i in range(0, len(inbound_nodes[0])):
            pre_layer_nm = inbound_nodes[0][i][0]
            pre_layer_idx = get_layer_index_by_layer_name(model_json, pre_layer_nm)
            layer_idxes.append(pre_layer_idx)
        return layer_idxes
    return []


def get_all_pre_related_layers(model_json):
    all_pre_related_layers = set()
    pre_layers_set = set()
    model_layers_num = len(model_json['config']['layers'])
    for layer_idx in range(1, model_layers_num):
        pre_layers = get_precursor_layer_idxes(model_json, layer_idx)
        for pre_layer in pre_layers:
            if pre_layer not in pre_layers_set:
                pre_layers_set.add(pre_layer)
            else:
                all_pre_related_layers.add(pre_layer)
    return list(all_pre_related_layers)


def check_zero_weight(model):
    for layer_idx in range(0, len(model.layers)):
        layer = model.layers[layer_idx]
        weights = layer.get_weights()
        if len(weights) > 1:
            weight = weights[0]
            if np.any(weight == 0):
                print("I find zero in the weights of {0}-th layer ({1})".format(layer_idx, layer.name))
    print("no zero in weights")


def get_pre_weighted_layers(model, noweighted_layer_idx):
    for layer_idx in range(noweighted_layer_idx, 0, -1):
        layer = model.layers[layer_idx]
        weights = layer.get_weights()
        if len(weights) > 0:
            return layer_idx
    return noweighted_layer_idx


def get_target_onehot_label(target, class_num=10):
    one_hot = np.zeros(class_num, dtype=int)
    one_hot[target] = 1
    return one_hot


# Can only handle the mapping of 3-D featur_map to flattened 1-d array
# ret: [fm_idx, h_idx, w_idx]
def get_featuremap2dense_mapper(feature_maps):
    shape = feature_maps.shape
    ndim = feature_maps.ndim
    if (ndim == 3):
        mapper = []
        for fm_idx in range(0, shape[-1]):
            # print("----------fm------------"+str(fm_idx))
            # print(feature_maps[:,:,fm_idx])
            tmp_arr = np.zeros((shape[0], shape[1]), dtype=int)
            for h_idx in range(0, shape[0]):
                # print("----------h_idx------------" + str(h_idx))
                for w_idx in range(0, shape[1]):
                    index = h_idx * shape[1] * shape[2] + w_idx * shape[2] + fm_idx
                    # print(str(index)+" "+str(fm_fln[index]))
                    tmp_arr[h_idx][w_idx] = index
            mapper.append(tmp_arr)
        return np.array(mapper)
    print("ERROR, I can only handle 3d-featur_map!")
    return


def get_dense2featuremap_mapper(fmlayer_output_shape):
    shape = fmlayer_output_shape
    ndim = len(fmlayer_output_shape)
    # dense2fm_idxes = np.zeros((shape[0] * shape[1] * shape[2]), dtype=int)
    dense2fm_idxes = {}
    if (ndim == 3):
        for fm_idx in range(0, shape[-1]):
            for h_idx in range(0, shape[0]):
                for w_idx in range(0, shape[1]):
                    index = h_idx * shape[1] * shape[2] + w_idx * shape[2] + fm_idx
                    # dense2fm_idxes[index] = fm_idx
                    dense2fm_idxes[index] = [h_idx, w_idx, fm_idx]
        return dense2fm_idxes
    print("ERROR, I can only handle 3d-featur_map!")
    return



if __name__ == '__main__':
    print('Hello NNUtils')
    # model_path = './models/lenet5-relu'
    # model = load_model(model_path)
    # model = VGG19(weights='imagenet')
    # model = ResNet50(weights='imagenet')
    # model.summary()
    # model_json = json.loads(model.to_json())
    # get_pre_weighted_layers(model, 1)
