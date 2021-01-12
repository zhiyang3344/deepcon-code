from keras import backend, losses
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.models import load_model
from keras_applications.vgg19 import VGG19
from keras_applications.resnet50 import ResNet50
from keras import backend as K
import os
import data_util
from ioutil import *
import nn_util

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

IDENTITY_CONNECTED_LAYERS = ['ZeroPadding2D', 'BatchNormalization', 'Activation', 'MaxPooling2D', 'GlobalAveragePooling2D', 'AveragePooling2D']
MULYIPY_TO_ONE_LAYERS = ['Add']
MAXPOOLING2D = ['MaxPooling2D']
FLATTEN_LAYERS = ["Flatten"]
DENSE_LAYERS = ['Dense']
CONV_LAYERS = ['Conv2D']
INPUT_LAYER = ['InputLayer']

IMAGENET_MODELS = ['vgg19', 'resnet50']
MNIST_MODELS = ['lenet1', 'lenet4', 'lenet5']


def clip_image(model_name, hacked_image):
    # hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
    if model_name in ('vgg19', 'resnet50'):
        hacked_image[0][:, :, 0] = np.clip(hacked_image[0][:, :, 0], 0.0 - 103.939, 255.0 - 103.939)
        hacked_image[0][:, :, 1] = np.clip(hacked_image[0][:, :, 1], 0.0 - 116.779, 255.0 - 116.779)
        hacked_image[0][:, :, 2] = np.clip(hacked_image[0][:, :, 2], 0.0 - 123.680, 255.0 - 123.680)
    else:
        hacked_image[0] = np.clip(hacked_image[0], 0.0, 1.0)
    return hacked_image


def target_fgsm(model, image, y_target, epsilons=[0.1]):
    model_name = model.name
    target_label = np.argmax(y_target)
    model_input_layer = model.layers[0].input
    y_pred = model.output
    loss = losses.categorical_crossentropy(y_target, y_pred)
    gradient_func = K.gradients(loss, model_input_layer)
    functor = K.function([model_input_layer, K.learning_phase()], gradient_func)

    # attacked_image=np.copy(image)
    # gradient = functor([attacked_image, 0])[0]
    for eps in epsilons:
        attacked_image = np.copy(image)
        for step in range(0, 200):
            gradient = functor([attacked_image, 0])[0]
            attacked_image = attacked_image - np.sign(gradient) * eps
            attacked_image = clip_image(model_name, attacked_image)
            prediction = model.predict(attacked_image)
            attacked_label = np.argmax(prediction[0])
            if step % 10 == 0:
                print('eps:', eps, 'step:', step, 'target_label msp:', prediction[0][target_label])
            if attacked_label == target_label and prediction[0][target_label] > 0.6:
                return True, attacked_image
    return False, attacked_image


def generate_attack_imgs(model, target_label, x_test, y_test, img_ids):
    model_name = model.name
    attack_save_path = './outputs/fgsm_attack/{0}/'.format(model_name)
    linspace = 4
    epsilons = np.linspace(0, 0.2, num=linspace + 1)[1:]
    target_onehot = nn_util.get_target_onehot_label(target_label, y_test[0].shape[-1])
    for img_idx in range(0, len(x_test)):
        img_id = img_ids[img_idx]
        org_pred = np.argmax(model.predict(x_test[img_idx:img_idx + 1]))
        y_true = np.argmax(y_test[img_idx])
        if org_pred != y_true:
            continue
        # target attack
        img_file = attack_save_path + '/attack_to_{2}/imgs/imgid_{0}_from_{1}_to_{2}.pkl'.format(img_id, org_pred,
                                                                                                 target_label)
        flag, attacked_img = target_fgsm(model, x_test[img_idx:img_idx + 1], target_onehot, epsilons)
        if flag:
            print("attack successful! image id: {0}".format(img_id))
            write_pkl_to_file(attacked_img, img_file)
    attacked_dir = attack_save_path + '/attack_to_{0}/imgs'.format(target_label)
    return attacked_dir


def get_one_class_mnist_test_examples(class_label, model):
    (_, _), (x_test, y_test) = data_util.get_mnist_data()
    one_class_mnist_test_examples = []
    y_preds = model.predict(x_test)
    for idx in range(0, len(y_test)):
        if np.argmax(y_test[idx]) == class_label:
            if np.argmax(y_preds[idx]) == class_label:
                one_class_mnist_test_examples.append(x_test[idx])
    return np.array(one_class_mnist_test_examples)


def get_seedimg_infos_by_attackedimgs_dir(model_name, attack_img_dir, is_return_seedimgs=False,
                                          is_return_attackedimgs=False):
    list = os.listdir(attack_img_dir)
    seed_imgs = []
    attacked_imgs = []
    seed_img_infos = []
    if model_name in MNIST_MODELS:
        _, (x_test, y_test) = data_util.get_mnist_data()
    for i in range(0, len(list)):
        img_infos = list[i].split('_')
        seed_img_id = int(img_infos[1])
        seed_img_class = int(img_infos[3])
        seed_img_infos.append([seed_img_id, seed_img_class])

        if is_return_seedimgs:
            if model_name in IMAGENET_MODELS:
                seed_img_id = "%08d" % (seed_img_id)
                seed_img_file = "./inputs/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_" + str(seed_img_id) + ".JPEG"
                seed_img = image.load_img(seed_img_file, target_size=(224, 224))
                seed_img = image.img_to_array(seed_img)
                seed_imgs.append(seed_img)
            elif model_name in MNIST_MODELS:
                seed_img = x_test[seed_img_id]
                seed_imgs.append(seed_img)

        if is_return_attackedimgs:
            file = os.path.join(attack_img_dir, list[i])
            attacked_img = read_pkl_from_file(file)[0]
            attacked_imgs.append(attacked_img)
    if is_return_seedimgs:
        seed_imgs = np.array(seed_imgs)
        if model_name in IMAGENET_MODELS:
            seed_imgs = imagenet_utils.preprocess_input(seed_imgs)
    if is_return_attackedimgs:
        attacked_imgs = np.array(attacked_imgs)

    return seed_img_infos, seed_imgs, attacked_imgs


def output_contributions(model, target_label, batch_size=20, is_auto_clear_session=True):
    from nn_contribution_util import output_contributions
    model_name = model.name
    attack_save_path = './outputs/fgsm_attack/{0}/'.format(model_name)
    attack_img_dir = attack_save_path + '/attack_to_{0}/imgs/'.format(target_label)

    seed_img_infos, seed_imgs, attacked_imgs = get_seedimg_infos_by_attackedimgs_dir(model_name, attack_img_dir, True, True)

    attack_contris_dir = attack_save_path + "/attack_to_{0}/attack_contris/".format(target_label)
    seed_contris_dir = attack_save_path + "/attack_to_{0}/seed_contris/".format(target_label)
    for img_idx in range(0, len(attacked_imgs), batch_size):
        batch_attacked_imgs = attacked_imgs[img_idx:img_idx + batch_size]
        batch_attacked_contris, _ = output_contributions(model, batch_attacked_imgs, is_auto_clear_session=is_auto_clear_session)
        for idx in range(img_idx, img_idx + len(batch_attacked_imgs)):
            org_img_id, src_label = seed_img_infos[idx][0], seed_img_infos[idx][1]
            attack_contris_file = attack_contris_dir + "/imgid_{0}_from_{1}_to_{2}_contris.pkl".format(org_img_id, src_label, target_label)
            batch_attacked_contri = batch_attacked_contris[idx - img_idx]
            write_pkl_to_file(batch_attacked_contri, attack_contris_file, 'wb')

        batch_seed_imgs = seed_imgs[img_idx:img_idx + batch_size]
        batch_seed_img_contris, _ = output_contributions(model, batch_seed_imgs, is_auto_clear_session=is_auto_clear_session)
        for idx in range(img_idx, img_idx + len(batch_seed_imgs)):
            org_img_id, src_label = seed_img_infos[idx][0], seed_img_infos[idx][1]
            seed_contris_file = seed_contris_dir + "/imgid_{0}_from_{1}_contris.pkl".format(org_img_id, src_label)
            batch_seed_img_contri = batch_seed_img_contris[idx - img_idx]
            write_pkl_to_file(batch_seed_img_contri, seed_contris_file, 'wb')

    return attack_contris_dir, seed_contris_dir


def output_backprop_dfgs(model, target_label, threshold=0.75, batch_size=20):
    from dfg_util import extract_covered_dfgs, aggregate_graphs, gt_scaled_t
    model_name = model.name
    model_json = json.loads(model.to_json())
    attack_save_path = './outputs/fgsm_attack/{0}/'.format(model_name)
    attack_img_dir = attack_save_path + '/attack_to_{0}/imgs/'.format(target_label)
    seed_img_infos, seed_imgs, attacked_imgs = get_seedimg_infos_by_attackedimgs_dir(model_name, attack_img_dir, True,
                                                                                     True)

    attack_contris_dir = attack_save_path + "/attack_to_{0}/attack_contris/".format(target_label)
    seed_contris_dir = attack_save_path + "/attack_to_{0}/seed_contris/".format(target_label)

    attack_dfgs_dir = attack_save_path + "/attack_to_{0}/attack_dfgs/".format(target_label)
    seed_dfgs_dir = attack_save_path + "/attack_to_{0}/seed_dfgs/".format(target_label)

    for img_idx in range(0, len(attacked_imgs), batch_size):
        batch_attacked_imgs = attacked_imgs[img_idx:img_idx + batch_size]
        batch_attack_ns = nn_util.get_all_layers_out_vals(model, batch_attacked_imgs)

        batch_seed_imgs = seed_imgs[img_idx:img_idx + batch_size]
        batch_ssed_ns = nn_util.get_all_layers_out_vals(model, batch_seed_imgs)

        batch_attacked_contris = []
        batch_seed_contris = []
        for idx in range(img_idx, img_idx + len(batch_attacked_imgs)):
            org_img_id, src_label = seed_img_infos[idx]
            attacked_contris_file = attack_contris_dir + "/imgid_{0}_from_{1}_to_{2}_contris.pkl".format(org_img_id,
                                                                                                         src_label,
                                                                                                         target_label)
            attacked_contri = read_pkl_from_file(attacked_contris_file)
            batch_attacked_contris.append(attacked_contri)

            seed_contris_file = seed_contris_dir + "/imgid_{0}_from_{1}_contris.pkl".format(org_img_id, src_label)
            seed_contri = read_pkl_from_file(seed_contris_file)
            batch_seed_contris.append(seed_contri)

        _, batch_attack_covered_dfgs = extract_covered_dfgs(model_json, batch_attacked_contris, batch_attack_ns,
                                                            gt_scaled_t, threshold)
        _, batch_seed_covered_dfgs = extract_covered_dfgs(model_json, batch_seed_contris, batch_ssed_ns, gt_scaled_t,
                                                          threshold)

        for idx in range(img_idx, img_idx + len(batch_attacked_imgs)):
            org_img_id, src_label = seed_img_infos[idx]
            attacked_dfg_file = attack_dfgs_dir + "/imgid_{0}_from_{1}_to_{2}_t_{3}_dfg.pkl".format(org_img_id,
                                                                                                    src_label,
                                                                                                    target_label,
                                                                                                    threshold)
            attacked_dfg = batch_attack_covered_dfgs[idx - img_idx]
            write_pkl_to_file(attacked_dfg, attacked_dfg_file, 'wb')

            seed_dfg_file = seed_dfgs_dir + "/imgid_{0}_from_{1}_t_{2}_dfg.pkl".format(org_img_id, src_label, threshold)
            seed_dfg = batch_seed_covered_dfgs[idx - img_idx]
            write_pkl_to_file(seed_dfg, seed_dfg_file, 'wb')

    return attack_dfgs_dir, seed_dfgs_dir


# def caculate_similarty():
#     from DFGUtils import caculate_similarity_of_two_dfgs
#
#     model_path = './models/lenet5-relu'
#     model_name = 'lenet5'
#     model = load_model(model_path)
#
#     threshold = 0.75
#     src_label = 8
#     target_label = 1
#
#     attack_save_path = './outputs/fgsm_attack/{0}/'.format(model_name) + "{0}/".format(src_label)
#     attack_covered_dfgs_file = attack_save_path + "/dfg/to_{0}_with_t_{1}_dfgs.pkl".format(target_label, threshold)
#     attack_covered_dfgs = read_pkls_from_file(attack_covered_dfgs_file)
#     org_covered_dfgs_file = attack_save_path + "/dfg/org_with_t_{0}_dfgs.pkl".format(threshold)
#     org_covered_dfgs = read_pkls_from_file(org_covered_dfgs_file)
#     # single_dfg_similarity, single_each_layer_similarity = caculate_similarity_of_two_dfgs(model, org_covered_dfgs[0],
#     #                                                                         attack_covered_dfgs[0])
#
#     org_aggregated_dfg_file = attack_save_path + "/dfg/org_with_t_{0}.pkl".format(threshold)
#     attack_aggregated_dfg_file = attack_save_path + "/dfg/to_{0}_with_t_{1}.pkl".format(target_label, threshold)
#     org_aggregated_dfg = read_pkl_from_file(org_aggregated_dfg_file)
#     attack_aggregated_dfg = read_pkl_from_file(attack_aggregated_dfg_file)
#     dfg_similarity, each_layer_similarity = caculate_similarity_of_two_dfgs(model, org_aggregated_dfg,
#                                                                             attack_aggregated_dfg)
#     return dfg_similarity, each_layer_similarity


def compute_each_img_neuron_states(model, all_layers_out_vals, t_func, *args):
    from dfg_util import mean_by_last_axis, arg_max_flags
    layers_count = len(model.layers)
    each_img_neuron_states = [[None for layer_idx in range(0, layers_count)] for img_idx in
                              range(0, len(all_layers_out_vals[0]))]
    for layer_idx in range(0, layers_count):
        for img_idx in range(0, len(all_layers_out_vals[layer_idx])):
            print("processing {0}-th image in {1}-th layer".format(img_idx, layer_idx))
            layer_output_of_single_img = all_layers_out_vals[layer_idx][img_idx]
            layer_output_of_single_img = mean_by_last_axis(layer_output_of_single_img)
            if layer_idx == layers_count - 1:
                neuron_activated_flags = np.array(arg_max_flags(layer_output_of_single_img))
            else:
                neuron_activated_flags = np.array(t_func(layer_output_of_single_img, args))
            each_img_neuron_states[img_idx][layer_idx] = neuron_activated_flags
    return each_img_neuron_states


# def nc_output_dfgs():
#     from dfg_util import gt_scaled_t, get_node_name
#     from nn_contribution_util import get_featuremap2dense_mapper
#     model_path = './models/lenet5-relu'
#     model_name = 'lenet5'
#     model = load_model(model_path)
#     model.summary()
#     model_json = json.loads(model.to_json())
#     output_shapes = nn_util.get_dnn_each_layer_output_shape(model)
#
#     threshold = 0.5
#     src_label = 8
#     target_label = 1
#     attack_save_path = './outputs/fgsm_attack/{0}/'.format(model_name) + "{0}/".format(src_label)
#
#     attack_layers_outs_file = attack_save_path + "/layer_outs/attack_to_{0}_layer_outs.pkl".format(target_label)
#     org_layers_outs_file = attack_save_path + "/layer_outs/org_layer_outs.pkl"
#     attack_layers_outs = read_pkls_from_file(attack_layers_outs_file)
#     org_layers_outs = read_pkls_from_file(org_layers_outs_file)
#
#     attack_neuron_states = compute_each_img_neuron_states(model, attack_layers_outs, gt_scaled_t, threshold)
#     org_neuron_states = compute_each_img_neuron_states(model, org_layers_outs, gt_scaled_t, threshold)
#
#     def construct_dfg_through_act_neurons(layers_outs, neuron_states):
#         model_layers_num = len(model_json['config']['layers'])
#         output_shapes = nn_util.get_dnn_each_layer_output_shape(model)
#         all_dfgs=[]
#         for img_idx in range(0,len(neuron_states)):
#             img_dfg={}
#             for layer_idx in range(1, model_layers_num):
#                 cur_class_name = model_json["config"]["layers"][layer_idx]["class_name"]
#                 pre_node_num = output_shapes[layer_idx - 1][-1]
#                 cur_node_num = output_shapes[layer_idx][-1]
#                 if cur_class_name in ['Dense', 'Conv2D']:
#                     for pre_node_idx in range(0, pre_node_num):
#                         if neuron_states[img_idx][layer_idx-1][pre_node_idx]>0:
#                             pre_node = get_node_name(layer_idx-1, pre_node_idx)
#                             cur_nodes=[]
#                             for cur_node_idx in range(0, cur_node_num):
#                                 if neuron_states[img_idx][layer_idx][cur_node_idx]>0:
#                                     cur_node=get_node_name(layer_idx, cur_node_idx)
#                                     cur_nodes.append(cur_node)
#                             if len(cur_nodes)>0:
#                                 img_dfg[pre_node]=cur_nodes
#                 elif cur_class_name in ['MaxPooling2D']:
#                     for pre_node_idx in range(0, pre_node_num):
#                         cur_node_idx = pre_node_idx
#                         if neuron_states[img_idx][layer_idx - 1][pre_node_idx] > 0:
#                             if neuron_states[img_idx][layer_idx][cur_node_idx] > 0:
#                                 pre_node = get_node_name(layer_idx - 1, pre_node_idx)
#                                 cur_node = get_node_name(layer_idx, cur_node_idx)
#                                 img_dfg[pre_node]=[cur_node]
#                 elif cur_class_name in ['Flatten']:
#                     pre_layer_feature_maps = layers_outs[layer_idx - 1][img_idx]
#                     flatten_map_arr = get_featuremap2dense_mapper(pre_layer_feature_maps)
#                     pre_node_num = output_shapes[layer_idx - 1][-1]
#                     for pre_node_idx in range(0, pre_node_num):
#                         if neuron_states[img_idx][layer_idx - 1][pre_node_idx] <= 0:
#                             continue
#                         pre_node = get_node_name(layer_idx - 1, pre_node_idx)
#                         cur_nodes=[]
#                         for w_idx in range(0, flatten_map_arr.shape[1]):
#                             for h_idx in range(0, flatten_map_arr.shape[2]):
#                                 if pre_layer_feature_maps[..., pre_node_idx][w_idx][h_idx] == 0:
#                                     continue
#                                 cur_node_idx = int(flatten_map_arr[pre_node_idx][w_idx][h_idx])
#                                 cur_node = get_node_name(layer_idx, cur_node_idx)
#                                 cur_nodes.append(cur_node)
#                         if len(cur_nodes) > 0:
#                             img_dfg[pre_node] = cur_nodes
#             all_dfgs.append(img_dfg)
#         return all_dfgs
#
#     org_dfgs = construct_dfg_through_act_neurons(org_layers_outs, org_neuron_states)
#     attack_dfgs = construct_dfg_through_act_neurons(attack_layers_outs, attack_neuron_states)
#
#     nc_attack_covered_dfgs_file = attack_save_path + "/nc_dfgs/to_{0}_with_t_{1}_dfgs.pkl".format(target_label, threshold)
#     nc_org_covered_dfgs_file = attack_save_path + "/nc_dfgs/org_with_t_{0}_dfgs.pkl".format(threshold)
#     write_pkls_to_file(org_dfgs, nc_org_covered_dfgs_file)
#     write_pkls_to_file(attack_dfgs, nc_attack_covered_dfgs_file)
#
#     return org_dfgs, attack_dfgs


# def nob_output_dfgs():
#     from DFGUtils import calculate_each_layer_covered_node_and_edge, mean_by_last_axis, get_node_name, get_node_idx_by_node_name
#     from OutputDNNPreds import get_featuremap2dense_mapper
#
#     model_path = './models/lenet5-relu'
#     model_name = 'lenet5'
#     model = load_model(model_path)
#     model.summary()
#     model_json = json.loads(model.to_json())
#     output_shapes = NNUtils.get_dnn_each_layer_output_shape(model)
#
#     threshold = 0.75
#     src_label = 8
#     target_label = 1
#
#     model_layers_num = len(model_json['config']['layers'])
#     attack_save_path = './outputs/fgsm_attack/{0}/'.format(model_name) + "{0}/".format(src_label)
#
#     attack_covered_dfgs_file = attack_save_path + "/dfgs/to_{0}_with_t_{1}_dfgs.pkl".format(target_label, threshold)
#     attack_covered_dfgs = read_pkls_from_file(attack_covered_dfgs_file)
#     org_covered_dfgs_file = attack_save_path + "/dfgs/org_with_t_{0}_dfgs.pkl".format(threshold)
#     org_covered_dfgs = read_pkls_from_file(org_covered_dfgs_file)
#
#     attacked_layers_outs_file = attack_save_path + "/layer_outs/attack_to_{0}_layer_outs.pkl".format(target_label)
#     org_layers_outs_file = attack_save_path + "/layer_outs/org_layer_outs.pkl"
#     attacked_layers_outs = read_pkls_from_file(attacked_layers_outs_file)
#     org_layers_outs = read_pkls_from_file(org_layers_outs_file)
#
#     def get_arg_k_max(layer_output, k):
#         sort_idx = np.argsort(layer_output)
#         top_k_idxs = sort_idx[len(sort_idx) - k:len(sort_idx)]
#         return top_k_idxs
#
#     # 取与con-cov.方法，相同数量的连接（两个神经元激活，则认为它们之间的连接激活）
#     # neuron-output-based简写为nob
#     # Flatten之前：从浅层→深层提取
#     # Flatten之后，从深层→浅层提取
#     def extract_same_mount_cons(in_cons_count, layers_outs):
#         nob_activated_nodes = {}
#         nob_activated_cons = {}
#         flatten_idx = -1
#         for layer_idx in range(0, model_layers_num):
#             nob_activated_nodes[layer_idx] = set()
#             cur_class_name = model_json["config"]["layers"][layer_idx]["class_name"]
#             if cur_class_name in ['Flatten']:
#                 flatten_idx = layer_idx
#         nob_activated_nodes[0].add(get_node_name(0, 0))
#
#         #浅层→深层
#         # for layer_idx in range(1, model_layers_num):
#         for layer_idx in range(1, flatten_idx+1):
#             cur_class_name = model_json["config"]["layers"][layer_idx]["class_name"]
#             if layer_idx == model_layers_num - 1:
#                 pred_idx = np.argmax(layers_outs[layer_idx][img_idx])
#                 cur_node = get_node_name(layer_idx, pred_idx)
#                 nob_activated_nodes[layer_idx].add(cur_node)
#
#                 # 构造cons
#                 for pre_node in nob_activated_nodes[layer_idx - 1]:
#                     if pre_node not in nob_activated_cons:
#                         nob_activated_cons[pre_node] = []
#                     nob_activated_cons[pre_node].append(cur_node)
#
#             elif cur_class_name in ['Dense', 'Conv2D']:
#             # if cur_class_name in ['Dense', 'Conv2D']:
#                 curlayer_layer_output = layers_outs[layer_idx][img_idx]
#                 curlayer_layer_output = mean_by_last_axis(curlayer_layer_output)
#
#                 curlayer_neuron_count = int(in_cons_count[layer_idx] / len(nob_activated_nodes[layer_idx - 1]))
#                 topk_idxs = get_arg_k_max(curlayer_layer_output, curlayer_neuron_count)
#                 for topk_idx in topk_idxs:
#                     cur_node = get_node_name(layer_idx, topk_idx)
#                     nob_activated_nodes[layer_idx].add(cur_node)
#
#                     # 构造cons
#                     for pre_node in nob_activated_nodes[layer_idx - 1]:
#                         if pre_node not in nob_activated_cons:
#                             nob_activated_cons[pre_node] = []
#                         nob_activated_cons[pre_node].append(cur_node)
#
#             elif cur_class_name in ['MaxPooling2D']:
#                 pre_layer_activated_nodes = nob_activated_nodes[layer_idx - 1]
#                 for pre_node in pre_layer_activated_nodes:
#                     pre_node_idx = get_node_idx_by_node_name(pre_node)
#                     cur_node = get_node_name(layer_idx, pre_node_idx)
#                     nob_activated_nodes[layer_idx].add(cur_node)
#                     # 构造cons
#                     if pre_node not in nob_activated_cons:
#                         nob_activated_cons[pre_node] = []
#                     nob_activated_cons[pre_node].append(cur_node)
#
#             elif cur_class_name in ['Flatten']:
#                 pre_layer_feature_maps = layers_outs[layer_idx - 1][img_idx]
#                 flatten_map_arr = get_featuremap2dense_mapper(pre_layer_feature_maps)
#                 pre_node_num = output_shapes[layer_idx - 1][-1]
#                 for pre_node_idx in range(0, pre_node_num):
#                     pre_node = get_node_name(layer_idx - 1, pre_node_idx)
#                     if pre_node not in nob_activated_nodes[layer_idx - 1]:
#                         continue
#                     if np.all(pre_layer_feature_maps[..., pre_node_idx] == 0):
#                         continue
#                     for w_idx in range(0, flatten_map_arr.shape[1]):
#                         for h_idx in range(0, flatten_map_arr.shape[2]):
#                             if pre_layer_feature_maps[..., pre_node_idx][w_idx][h_idx] == 0:
#                                 continue
#                             cur_node_idx = int(flatten_map_arr[pre_node_idx][w_idx][h_idx])
#                             cur_node = get_node_name(layer_idx, cur_node_idx)
#                             nob_activated_nodes[layer_idx].add(cur_node)
#
#                             # 构造cons
#                             if pre_node not in nob_activated_cons:
#                                 nob_activated_cons[pre_node] = []
#                             nob_activated_cons[pre_node].append(cur_node)
#         # 深层→浅层
#         for layer_idx in range(model_layers_num-1, flatten_idx, -1):
#             cur_class_name = model_json["config"]["layers"][layer_idx]["class_name"]
#             pre_class_name = model_json["config"]["layers"][layer_idx-1]["class_name"]
#             prelayer_layer_output = mean_by_last_axis(layers_outs[layer_idx - 1][img_idx])
#             if layer_idx == model_layers_num - 1:
#                 cur_idx = np.argmax(layers_outs[layer_idx][img_idx])
#                 cur_node = get_node_name(layer_idx, cur_idx)
#                 nob_activated_nodes[layer_idx].add(cur_node)
#
#                 prelayer_neuron_count = in_cons_count[layer_idx]
#                 prelayer_topk_idxs = get_arg_k_max(prelayer_layer_output, prelayer_neuron_count)
#                 for prelayer_topk_idx in prelayer_topk_idxs:
#                     pre_node = get_node_name(layer_idx-1, prelayer_topk_idx)
#                     nob_activated_nodes[layer_idx-1].add(pre_node)
#                     # 构造cons
#                     for cur_node in nob_activated_nodes[layer_idx]:
#                         if pre_node not in nob_activated_cons:
#                             nob_activated_cons[pre_node] = []
#                         nob_activated_cons[pre_node].append(cur_node)
#
#             elif cur_class_name in ['Dense', 'Conv2D']:
#                 prelayer_neuron_count = int(in_cons_count[layer_idx] / len(nob_activated_nodes[layer_idx]))
#                 prelayer_topk_idxs = get_arg_k_max(prelayer_layer_output, prelayer_neuron_count)
#                 for prelayer_topk_idx in prelayer_topk_idxs:
#                     pre_node = get_node_name(layer_idx-1, prelayer_topk_idx)
#                     if pre_class_name in ['Flatten']:
#                         if pre_node not in nob_activated_nodes[layer_idx - 1]:
#                             continue
#                     nob_activated_nodes[layer_idx-1].add(pre_node)
#
#                     # 构造cons
#                     for cur_node in nob_activated_nodes[layer_idx]:
#                         if pre_node not in nob_activated_cons:
#                             nob_activated_cons[pre_node] = []
#                         nob_activated_cons[pre_node].append(cur_node)
#
#         return nob_activated_nodes, nob_activated_cons
#
#     attack_nob_covered_dfgs = []
#     org_nob_covered_dfgs = []
#     for img_idx in range(0, len(attack_covered_dfgs)):
#         attack_dfg = attack_covered_dfgs[img_idx]
#         attack_nodes_count, attack_cons_count = calculate_each_layer_covered_node_and_edge(model, attack_dfg, [], [], [])
#         attack_nob_activated_nodes, attack_nob_activated_cons = extract_same_mount_cons(attack_cons_count, attacked_layers_outs)
#         attack_nob_covered_dfgs.append(attack_nob_activated_cons)
#
#         org_dfg = org_covered_dfgs[img_idx]
#         org_count, org_cons_count = calculate_each_layer_covered_node_and_edge(model, org_dfg, [], [],[])
#         org_nob_activated_nodes, org_nob_activated_cons = extract_same_mount_cons(org_cons_count, org_layers_outs)
#         org_nob_covered_dfgs.append(org_nob_activated_cons)
#
#
#     nc_attack_covered_dfgs_file = attack_save_path + "/backprop_nc_dfgs/to_{0}_with_t_{1}_dfgs.pkl".format(target_label, threshold)
#     nc_org_covered_dfgs_file = attack_save_path + "/backprop_nc_dfgs/org_with_t_{0}_dfgs.pkl".format(threshold)
#     write_pkls_to_file(org_nob_covered_dfgs, nc_org_covered_dfgs_file)
#     write_pkls_to_file(attack_nob_covered_dfgs, nc_attack_covered_dfgs_file)
#
#     return org_nob_covered_dfgs, attack_nob_covered_dfgs


def construct_nc_dfg_through_backprop_dfg(model, layers_outs, backprop_dfgs):
    from dfg_util import mean_by_last_axis, get_node_name, get_node_idx_by_node_name
    from nn_contribution_util import get_dense2featuremap_mapper
    model_json = json.loads(model.to_json())
    output_shapes = nn_util.get_dnn_each_layer_output_shape(model)
    model_layers_num = len(model_json['config']['layers'])

    def get_arg_k_max(layer_output, k):
        sort_idx = np.argsort(layer_output)
        top_k_idxs = sort_idx[len(sort_idx) - k:len(sort_idx)]
        return top_k_idxs

    def count_each_actneurons(backprop_dfg):
        model_layers_num = len(model_json['config']['layers'])
        each_actneuron_count = [0 for i in range(0, model_layers_num)]
        for layer_idx in range(0, model_layers_num - 1):
            cur_node_num = output_shapes[layer_idx][-1]
            for cur_node_idx in range(0, cur_node_num):
                cur_node = get_node_name(layer_idx, cur_node_idx)
                if cur_node in backprop_dfg:
                    each_actneuron_count[layer_idx] += 1
        each_actneuron_count[model_layers_num - 1] = 1
        return each_actneuron_count

    activated_all_img_nodes = []
    activated_dfgs = []
    for img_idx in range(0, len(backprop_dfgs)):
        activated_nodes = [set() for i in range(0, model_layers_num)]
        activated_dfg = {}
        backprop_dfg = backprop_dfgs[img_idx]
        each_actneuron_count = count_each_actneurons(backprop_dfg)

        curlayer_layer_output = mean_by_last_axis(layers_outs[0][img_idx])
        topk_idxs = get_arg_k_max(curlayer_layer_output, each_actneuron_count[0])
        for topk_idx in topk_idxs:
            cur_node = get_node_name(0, topk_idx)
            activated_nodes[0].add(cur_node)

        for layer_idx in range(1, model_layers_num):
            cur_class_name = model_json["config"]["layers"][layer_idx]["class_name"]
            curlayer_layer_output = mean_by_last_axis(layers_outs[layer_idx][img_idx])
            curlayer_neuron_count = each_actneuron_count[layer_idx]
            pre_layer_idxes = nn_util.get_precursor_layer_idxes(model_json, layer_idx)
            if cur_class_name in ['Dense', 'Conv2D']:
                topk_idxs = get_arg_k_max(curlayer_layer_output, curlayer_neuron_count)
                for topk_idx in topk_idxs:
                    cur_node = get_node_name(layer_idx, topk_idx)
                    activated_nodes[layer_idx].add(cur_node)
                    pre_layer_idx = pre_layer_idxes[0]
                    for pre_node in activated_nodes[pre_layer_idx]:
                        if pre_node not in activated_dfg:
                            activated_dfg[pre_node] = []
                        activated_dfg[pre_node].append(cur_node)
            elif cur_class_name in IDENTITY_CONNECTED_LAYERS:
                # elif cur_class_name in ['MaxPooling2D']:
                pre_layer_idx = pre_layer_idxes[0]
                pre_layer_activated_nodes = activated_nodes[pre_layer_idx]
                for pre_node in pre_layer_activated_nodes:
                    pre_node_idx = get_node_idx_by_node_name(pre_node)
                    cur_node = get_node_name(layer_idx, pre_node_idx)
                    activated_nodes[layer_idx].add(cur_node)
                    activated_dfg[pre_node] = [cur_node]
            elif cur_class_name in MULYIPY_TO_ONE_LAYERS:
                for pre_layer_idx in pre_layer_idxes:
                    curlayer_topk_idxs = get_arg_k_max(curlayer_layer_output, curlayer_neuron_count)
                    for curlayer_topk_idx in curlayer_topk_idxs:
                        cur_node = get_node_name(layer_idx, curlayer_topk_idx)
                        activated_nodes[layer_idx].add(cur_node)
                        # 构造dfg
                        pre_node = get_node_name(pre_layer_idx, curlayer_topk_idx)
                        if pre_node in activated_nodes[pre_layer_idx]:
                            if pre_node not in activated_dfg:
                                activated_dfg[pre_node] = []
                            activated_dfg[pre_node].append(cur_node)

            elif cur_class_name in ['Flatten']:
                pre_layer_idx = pre_layer_idxes[0]
                pre_layer_outs = layers_outs[pre_layer_idx][img_idx]
                dense2featuremap_mapper = get_dense2featuremap_mapper(pre_layer_outs.shape)
                curlayer_neuron_count = each_actneuron_count[layer_idx]
                topk_idxs = get_arg_k_max(curlayer_layer_output, curlayer_neuron_count)
                for topk_idx in topk_idxs:
                    cur_node = get_node_name(layer_idx, topk_idx)
                    h_idx, w_idx, pre_node_idx = dense2featuremap_mapper[topk_idx]
                    pre_node = get_node_name(layer_idx - 1, pre_node_idx)
                    if pre_node not in activated_nodes[layer_idx - 1]:
                        continue
                    activated_nodes[layer_idx].add(cur_node)
                    if pre_node not in activated_dfg:
                        activated_dfg[pre_node] = []
                    activated_dfg[pre_node].append(cur_node)
        activated_all_img_nodes.append(activated_nodes)
        activated_dfgs.append(activated_dfg)
    return activated_all_img_nodes, activated_dfgs


# 使用neuron-output-based cover method输出与backpropagation cover method想同数量的neurons的dfg
def output_nc_dfgs_through_backprop_dfgs(model, target_label, threshold):
    model_name = model.name
    attack_save_path = './outputs/fgsm_attack/{0}/'.format(model_name)

    attack_dfgs_dir = attack_save_path + "/attack_to_{0}/attack_dfgs/".format(target_label)
    seed_dfgs_dir = attack_save_path + "/attack_to_{0}/seed_dfgs/".format(target_label)

    nc_attack_dfgs_dir = attack_save_path + "/attack_to_{0}/attack_ncdfgs/".format(target_label)
    nc_seed_dfgs_dir = attack_save_path + "/attack_to_{0}/seed_ncdfgs/".format(target_label)

    attack_img_dir = attack_save_path + '/attack_to_{0}/imgs/'.format(target_label)
    seed_img_infos, seed_imgs, attacked_imgs = get_seedimg_infos_by_attackedimgs_dir(model_name, attack_img_dir, True,
                                                                                     True)
    for img_idx in range(0, len(attacked_imgs)):
        org_img_id, src_label = seed_img_infos[img_idx]

        attacked_img = attacked_imgs[img_idx:img_idx + 1]
        attacked_n = nn_util.get_all_layers_out_vals(model, attacked_img)
        attacked_dfg_file = attack_dfgs_dir + "/imgid_{0}_from_{1}_to_{2}_t_{3}_dfg.pkl".format(org_img_id, src_label,
                                                                                                target_label, threshold)
        attacked_dfg = read_pkl_from_file(attacked_dfg_file)
        nc_attacked_nodes, nc_attacked_dfgs = construct_nc_dfg_through_backprop_dfg(model, attacked_n, [attacked_dfg])
        nc_attacked_dfg_file = nc_attack_dfgs_dir + "/imgid_{0}_from_{1}_to_{2}_t_{3}_dfg.pkl".format(org_img_id,
                                                                                                      src_label,
                                                                                                      target_label,
                                                                                                      threshold)
        write_pkl_to_file(nc_attacked_dfgs[0], nc_attacked_dfg_file, 'wb')

        seed_img = seed_imgs[img_idx:img_idx + 1]
        seed_n = nn_util.get_all_layers_out_vals(model, seed_img)
        seed_dfg_file = seed_dfgs_dir + "/imgid_{0}_from_{1}_t_{2}_dfg.pkl".format(org_img_id, src_label, threshold)
        seed_dfg = read_pkl_from_file(seed_dfg_file)
        nc_seed_nodes, nc_seed_dfgs = construct_nc_dfg_through_backprop_dfg(model, seed_n, [seed_dfg])
        nc_seed_dfg_file = nc_seed_dfgs_dir + "/imgid_{0}_from_{1}_t_{2}_dfg.pkl".format(org_img_id, src_label,
                                                                                         threshold)
        write_pkl_to_file(nc_seed_dfgs[0], nc_seed_dfg_file, 'wb')

    return nc_attack_dfgs_dir, nc_seed_dfgs_dir


def aggregate_each_clean_class_dfg(model, labels, threshold, batch_size, is_auto_clear_session=True):
    from nn_contribution_util import output_contributions
    from dfg_util import extract_covered_dfgs, aggregate_graphs, gt_scaled_t
    model_json = json.loads(model.to_json())
    model_name = model.name
    clean_path = './outputs/clean/{0}/'.format(model_name)
    class_dfgs_dir = clean_path + '/clean_classdfgs/'
    nc_class_dfgs_dir = clean_path + '/clean_ncclassdfgs/'

    for img_class in labels:
        src_label = img_class
        if model_name in ['vgg19', 'resnet50']:
            imgs, labels, img_ids = data_util.load_imagenet_train_data([img_class])
        elif model_name in ['lenet1', 'lenet4', 'lenet5']:
            imgs, labels, img_ids = data_util.load_mnist_tarin_data([img_class])
        else:
            raise ValueError('Unsupported model {0} !'.format(model_name))

        # 1. output contribution
        clean_contris_dir = clean_path + "/from_{0}_contris/".format(img_class)
        for img_idx in range(0, len(imgs), batch_size):
            print("Extract contributions on images[{0}:{1}] from total {2} images.".format(img_idx, img_idx + batch_size, len(imgs)))
            batch_imgs = imgs[img_idx:img_idx + batch_size]
            batch_contris, _ = output_contributions(model, batch_imgs, is_auto_clear_session=is_auto_clear_session)
            print("the length of batch_contris is {0}".format(len(batch_contris)))
            for idx in range(img_idx, img_idx + len(batch_imgs)):
                org_img_id = img_ids[idx]
                contris_file = clean_contris_dir + "/imgid_{0}_from_{1}_contris.pkl".format(org_img_id, src_label)
                print("write contris_file: {0}".format(contris_file))
                batch_contri = batch_contris[idx - img_idx]
                write_pkl_to_file(batch_contri, contris_file, 'wb')

        # 2. construct dfgs
        if is_auto_clear_session == True:
            model = nn_util.auto_clear_session_and_rebuild_model(model)
        clean_contris_dir = clean_path + "/from_{0}_contris/".format(img_class)
        clean_dfgs_dir = clean_path + "/from_{0}_dfgs/".format(img_class)
        nc_clean_dfgs_dir = clean_path + "/from_{0}_ncdfgs/".format(img_class)
        for img_idx in range(0, len(imgs), batch_size):
            print("Construct dfgs on images[{0}:{1}] from total {2} images.".format(img_idx, img_idx + batch_size, len(imgs)))
            batch_imgs = imgs[img_idx:img_idx + batch_size]
            batch_contris = []
            for idx in range(img_idx, img_idx + len(batch_imgs)):
                org_img_id = img_ids[idx]
                contris_file = clean_contris_dir + "/imgid_{0}_from_{1}_contris.pkl".format(org_img_id, src_label)
                batch_contri = read_pkl_from_file(contris_file)
                batch_contris.append(batch_contri)
            # 2.1 output deepcon-dfgs
            batch_ns = nn_util.get_all_layers_out_vals(model, batch_imgs)
            _, batch_dfgs = extract_covered_dfgs(model_json, batch_contris, batch_ns, gt_scaled_t, threshold)
            for idx in range(img_idx, img_idx + len(batch_dfgs)):
                org_img_id = img_ids[idx]
                dfg_file = clean_dfgs_dir + "/imgid_{0}_from_{1}_t_{2}_dfg.pkl".format(org_img_id, src_label, threshold)
                dfg = batch_dfgs[idx - img_idx]
                write_pkl_to_file(dfg, dfg_file, 'wb')

            # 2.2 output nc-dfgs
            nc_batch_nodes, nc_batch_dfgs = construct_nc_dfg_through_backprop_dfg(model, batch_ns, batch_dfgs)
            for idx in range(img_idx, img_idx + len(nc_batch_dfgs)):
                org_img_id = img_ids[idx]
                nc_dfg_file = nc_clean_dfgs_dir + "/imgid_{0}_from_{1}_t_{2}_ncdfg.pkl".format(org_img_id, src_label,
                                                                                               threshold)
                nc_dfg = nc_batch_dfgs[idx - img_idx]
                write_pkl_to_file(nc_dfg, nc_dfg_file, 'wb')

        # 3. aggerate class-dfg
        clean_dfgs_dir = clean_path + "/from_{0}_dfgs/".format(img_class)
        nc_clean_dfgs_dir = clean_path + "/from_{0}_ncdfgs/".format(img_class)

        clean_class_dfg = {}
        nc_clean_class_dfg = {}
        for img_idx in range(0, len(imgs)):
            print("Aggerate class-dfg over the {0}-th mage of total {1} images.".format(img_idx, len(imgs)))
            org_img_id = img_ids[img_idx]
            dfg_file = clean_dfgs_dir + "/imgid_{0}_from_{1}_t_{2}_dfg.pkl".format(org_img_id, src_label, threshold)
            dfg = read_pkl_from_file(dfg_file)
            clean_class_dfg = aggregate_graphs(clean_class_dfg, [dfg])

            nc_dfg_file = nc_clean_dfgs_dir + "/imgid_{0}_from_{1}_t_{2}_ncdfg.pkl".format(org_img_id, src_label,
                                                                                           threshold)
            nc_dfg = read_pkl_from_file(nc_dfg_file)
            nc_clean_class_dfg = aggregate_graphs(nc_clean_class_dfg, [nc_dfg])
        clean_class_dfg_file = class_dfgs_dir + "/from_{0}_classdfg.pkl".format(img_class)
        nc_clean_class_dfg_file = nc_class_dfgs_dir + "/from_{0}_ncclassdfg.pkl".format(img_class)
        write_pkl_to_file(clean_class_dfg, clean_class_dfg_file, 'wb')
        write_pkl_to_file(nc_clean_class_dfg, nc_clean_class_dfg_file, 'wb')


def calculate_overlap(model, target_labels):
    from ood_detection import calculate_intersection, calculate_overlap
    import matplotlib.pyplot as plt
    model_name = model.name
    attack_save_path = './outputs/fgsm_attack/{0}/'.format(model_name)

    clean_path = './outputs/clean/{0}/'.format(model_name)
    class_dfgs_dir = clean_path + '/clean_classdfgs/'
    nc_class_dfgs_dir = clean_path + '/clean_ncclassdfgs/'

    exclude_layers = []

    def get_dfgs_from_dir(dir):
        list = os.listdir(dir)
        dfgs = []
        img_infos = []
        for i in range(0, len(list)):
            dfg_file = os.path.join(dir, list[i])
            dfg = read_pkl_from_file(dfg_file)
            dfgs.append(dfg)

            dfg_infos = list[i].split('_')
            img_id = int(dfg_infos[1])
            if 'to' in dfg_infos:
                target = int(dfg_infos[5])
            else:
                target = int(dfg_infos[3])
            img_infos.append([img_id, target])
        return dfgs, img_infos

    def get_classdfgs_from_dir(dir):
        list = os.listdir(dir)
        classdfgs = {}
        for i in range(0, len(list)):
            classdfg_file = os.path.join(dir, list[i])
            classdfg = read_pkl_from_file(classdfg_file)
            classdfg_infos = list[i].split('_')
            label = int(classdfg_infos[1])
            classdfgs[label] = classdfg
        return classdfgs

    def calculate_auc(attack_dfgs_dirs, seed_dfgs_dirs, class_dfgs_dir):
        from sklearn import metrics
        from sklearn.metrics import auc
        all_y_trues = []
        all_scores = []
        for dir_idx in range(0, len(attack_dfgs_dirs)):
            attack_dfgs_dir = attack_dfgs_dirs[dir_idx]
            seed_dfgs_dir = seed_dfgs_dirs[dir_idx]
            attack_dfgs, attack_img_infos = get_dfgs_from_dir(attack_dfgs_dir)
            seed_dfgs, seed_img_infos = get_dfgs_from_dir(seed_dfgs_dir)
            class_dfgs = get_classdfgs_from_dir(class_dfgs_dir)
            true_label_and_scores = np.zeros((len(attack_dfgs) + len(seed_dfgs), 2))
            for attack_dfg_idx in range(0, len(attack_dfgs)):
                attack_dfg = attack_dfgs[attack_dfg_idx]
                attack_lable = attack_img_infos[attack_dfg_idx][1]
                class_dfg = class_dfgs[attack_lable]

                intersected_dfg = calculate_intersection(attack_dfg, class_dfg)
                dfg_similarity, each_layer_similarity = calculate_overlap(model, intersected_dfg, attack_dfg, exclude_layers)

                print("attack: ", attack_dfg_idx, dfg_similarity, each_layer_similarity)
                true_label_and_scores[attack_dfg_idx] = [1, 1 - dfg_similarity]

            print("\n")

            for seed_dfg_idx in range(0, len(seed_dfgs)):
                seed_dfg = seed_dfgs[seed_dfg_idx]
                seed_label = seed_img_infos[seed_dfg_idx][1]
                class_dfg = class_dfgs[seed_label]
                intersected_dfg = calculate_intersection(seed_dfg, class_dfg)
                dfg_similarity, each_layer_similarity = calculate_overlap(model, intersected_dfg, seed_dfg, exclude_layers)
                print("org: ", seed_dfg_idx + len(attack_dfgs), dfg_similarity, each_layer_similarity)
                true_label_and_scores[seed_dfg_idx + len(attack_dfgs)] = [0, 1 - dfg_similarity]
            y_trues = true_label_and_scores[::, 0]
            scores = true_label_and_scores[::, 1]
            all_y_trues += list(y_trues)
            all_scores += list(scores)
        all_y_trues = np.array(all_y_trues)
        all_scores = np.array(all_scores)
        fpr, tpr, thresholds = metrics.roc_curve(all_y_trues, all_scores)
        au_roc = auc(fpr, tpr)
        return fpr, tpr, thresholds, au_roc

    def plot_auroc(fpr, tpr, au_roc, label):
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr, 'k--', label=label + ' (AUC = {0:.2f})'.format(au_roc), lw=2)
        # plt.title('AU-ROC')
        plt.legend(loc="lower right")
        # plt.plot(fpr, tpr, marker='o')
        plt.show()

    nc_attack_dfgs_dirs = []
    nc_seed_dfgs_dirs = []
    attack_dfgs_dirs = []
    seed_dfgs_dirs = []
    for target_label in target_labels:
        attack_dfgs_dir = attack_save_path + "/attack_to_{0}/attack_dfgs/".format(target_label)
        seed_dfgs_dir = attack_save_path + "/attack_to_{0}/seed_dfgs/".format(target_label)
        nc_attack_dfgs_dir = attack_save_path + "/attack_to_{0}/attack_ncdfgs/".format(target_label)
        nc_seed_dfgs_dir = attack_save_path + "/attack_to_{0}/seed_ncdfgs/".format(target_label)
        nc_attack_dfgs_dirs.append(nc_attack_dfgs_dir)
        nc_seed_dfgs_dirs.append(nc_seed_dfgs_dir)
        attack_dfgs_dirs.append(attack_dfgs_dir)
        seed_dfgs_dirs.append(seed_dfgs_dir)

    nc_fpr, nc_tpr, nc_thresholds, nc_au_roc = calculate_auc(nc_attack_dfgs_dirs, nc_seed_dfgs_dirs, nc_class_dfgs_dir)
    backprop_fpr, backprop_tpr, backprop_thresholds, backprop_au_roc = calculate_auc(attack_dfgs_dirs, seed_dfgs_dirs, class_dfgs_dir)

    plot_auroc(nc_fpr, nc_tpr, nc_au_roc, 'Neuron-output-based covered method')
    plot_auroc(backprop_fpr, backprop_tpr, backprop_au_roc, 'Back-propogation covered method')

    print("backprop_fpr =", backprop_fpr.tolist())
    print("backprop_tpr = ", backprop_tpr.tolist())
    print("backprop_au_roc = ", backprop_au_roc)

    print("\n")
    print("nc_fpr = ", nc_fpr.tolist())
    print("nc_tpr = ", nc_tpr.tolist())
    print("nc_au_roc = ", nc_au_roc)
    return backprop_fpr, backprop_tpr, backprop_au_roc, nc_fpr, nc_tpr, nc_au_roc


if __name__ == "__main__":
    model_path = './models/lenet4-relu'
    model_name = 'lenet4'
    # model_path = './models/lenet5-relu'
    # model_name = 'lenet5'
    model = load_model(model_path)
    model.name = model_name
    src_labels = [i for i in range(0, 10)]
    target_labels = [i for i in range(0, 10)]
    # _, (x_test, y_test) = data_util.get_mnist_data()
    x_test, y_test, img_ids = data_util.load_mnist_test_data(src_labels)

    # model = VGG19(weights='imagenet', backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    # model_name = 'vgg19'
    # model = ResNet50(weights='imagenet', backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    # model_name = 'resnet50'
    # target_labels = [i for i in range(851, 856)]
    # src_labels = [i for i in range(851, 856)]
    # x_test, y_test, img_ids = data_util.load_imgnet_val_data(src_labels)

    threshold = 0.75
    batch_size = 100
    model.summary()

    aggregate_each_clean_class_dfg(model, list(set(src_labels + target_labels)), threshold, batch_size, True)

    for target_label in target_labels:
        generate_attack_imgs(model, target_label,  x_test, y_test, img_ids)

    for target_label in target_labels:
        output_contributions(model, target_label, batch_size)

    for target_label in target_labels:
        output_backprop_dfgs(model, target_label, threshold, batch_size)

    for target_label in target_labels:
        output_nc_dfgs_through_backprop_dfgs(model, target_label, threshold)

    calculate_overlap(model, target_labels)
