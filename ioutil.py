import os
import json
import pickle
import sys


def write_json_to_file(obj, file_name, mode='w'):
    file_dir = os.path.split(file_name)[0]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    jsObj = json.dumps(obj)
    with open(file_name, mode) as f:
        f.write(jsObj)
        f.write("\n")


def write_jsons_to_file(objs, file_name, mode='a+'):
    file_dir = os.path.split(file_name)[0]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file_name, mode) as f:
        for obj in objs:
            jsobj = json.dumps(obj)
            f.write(jsobj)
            f.write("\n")


def read_jsons_from_file(file_name, mode='r'):
    with open(file_name, mode) as f:
        lines = f.readlines()
        json_objs = []
        for line in lines:
            data = json.loads(line)
            json_objs.append(data)
        return json_objs


def read_json_from_file(file_name, mode='r'):
    with open(file_name, mode) as f:
        data = json.load(f)
        return data


def write_pkls_to_file(objs, file_name, mode='ab+'):
    file_dir = os.path.split(file_name)[0]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    df2 = open(file_name, mode)
    for obj in objs:
        pickle.dump(obj, df2)
    df2.close()


def write_pkl_to_file(obj, file_name, mode='wb'):
    file_dir = os.path.split(file_name)[0]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    df2 = open(file_name, mode)
    pickle.dump(obj, df2)
    df2.close()


def read_pkl_from_file(file_name, mode='rb'):
    with open(file_name, mode) as f:
        data = pickle.load(f)
        return data


def read_pkls_from_file(file_name, mode='rb'):
    res = []
    with open(file_name, mode) as f:
        while True:
            try:
                res.append(pickle.load(f))
            except:
                return res


# def convert_kpls_to_jsons(kpls_file, jsons_file):
#     objs = read_pkls_from_file(kpls_file)
#     write_jsons_to_file(objs, jsons_file, 'w')
#
#
# def convert_jsons_to_kpls(jsons_file, kpls_file):
#     objs = read_jsons_from_file(jsons_file)
#     write_pkls_to_file(objs, kpls_file, 'wb')


def tar_kpls_file(kpls_file):
    import time
    (filepath, only_kpl_filename_ext) = os.path.split(kpls_file)
    (kpl_filename_no_ext, extension) = os.path.splitext(kpls_file)
    only_filename = os.path.split(kpl_filename_no_ext)[-1]
    exec_str = "cd " + filepath + " && tar -zcvf " + only_filename + ".tar.gz " + only_kpl_filename_ext + " && rm " + only_kpl_filename_ext
    print("I will exec: ", exec_str)
    os.system(exec_str)
    time.sleep(0.1)


def print_err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


if __name__ == '__main__':
    print_err("test print_err")
    # data_type = 'imagenet'
    # model_name = 'resnet50'
    # threshold = 0
    # t_path = "gt_" + str(threshold)
    #
    # start_idx = 0
    # end_idx = 5000
    # batch_size = 20
    # # output_batch_size = 100
    # dfg_base_dir = "./outputs/" + data_type + "/" + model_name + "/dfgs/" + t_path + "/"
    # for idx in range(start_idx, end_idx, batch_size):
    #     tmp_st = idx
    #     tmp_et = idx + batch_size
    #     covered_dfgs_file = dfg_base_dir + "dfg_" + str(tmp_st) + "-" + str(tmp_et) + ".pkl"
    #     tar_kpls_file(covered_dfgs_file)
