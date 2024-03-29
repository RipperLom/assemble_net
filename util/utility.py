import os
import sys
import json
import logging
import traceback

import tensorflow as tf


def get_all_files(train_data_file):
    """
    get all files
    """
    train_file = []
    train_path = train_data_file
    if os.path.isdir(train_path):
        data_parts = os.listdir(train_path)
        for part in data_parts:
            train_file.append(os.path.join(train_path, part))
    else:
        train_file.append(train_path)
    return train_file


def merge_config(config, *argv):
    """
    merge multiple configs
    """
    cf = {}
    cf.update(config)
    for d in argv:
        cf.update(d)
    return cf


def import_object(module_py, class_str):
    """
    string to class
    """
    mpath, mfile = os.path.split(module_py)
    sys.path.append(mpath)
    module=__import__(mfile)
    try:
        return getattr(module, class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                (class_str, traceback.format_exception(*sys.exc_info())))

def clazz(config, str1, str2):
    return import_object(config[str1], config[str2])


def seq_length(sequence):
    """
    get sequence length
    for id-sequence, (N, S)
        or vector-sequence  (N, S, D)
    """
    if len(sequence.get_shape().as_list()) == 2:
        used = tf.sign(tf.abs(sequence))
    else:
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def get_cross_mask(seq1, seq2):
    """
    get matching matrix mask, for two sequences( id-sequences or vector-sequences)
    """
    length1 = seq_length(seq1)
    length2 = seq_length(seq2)
    max_len1 = tf.shape(seq1)[1]
    max_len2 = tf.shape(seq2)[1]
    ##for padding left
    mask1 = tf.sequence_mask(length1, max_len1, dtype=tf.int32)
    mask2 = tf.sequence_mask(length2, max_len2, dtype=tf.int32)
    cross_mask = tf.einsum('ij,ik->ijk', mask1, mask2)
    return cross_mask


def load_config(config_file):
    """
    load config
    """
    with open(config_file, "r") as f:
        try:
            conf = json.load(f)
        except Exception:
            logging.error("load json file %s error" % config_file)
    conf_dict = {}
    unused = [conf_dict.update(conf[k]) for k in conf]
    logging.debug("\n".join(
        ["%s=%s" % (u, conf_dict[u]) for u in conf_dict]))
    return conf_dict