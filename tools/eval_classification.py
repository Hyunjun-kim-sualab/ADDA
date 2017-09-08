import logging
import os
import time
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm

import adda


def format_array(arr):
    return '  '.join(['{:.3f}'.format(x) for x in arr])


@click.command()
@click.argument('dataset')
@click.argument('split')
@click.argument('model')
@click.argument('weights')
@click.option('--gpu', default='0')
def main(dataset, split, model, weights, gpu):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    dataset_name = dataset
    split_name = split
    dataset = adda.data.get_dataset(dataset, shuffle=False)
    split = getattr(dataset, split)
    model_fn = adda.models.get_model_fn(model)
    im, label = split.tf_ops(capacity=2)
    im = adda.models.preprocessing(im, model_fn)
    im_batch, label_batch = tf.train.batch([im, label], batch_size=1)
    print(label_batch)

    net, layers = model_fn(im_batch, is_training=False)
    print(tf.shape(net))
    net_1 = net#tf.nn.softmax(net, 1)
    net = tf.argmax(net, 1)
    # net_1 = tf.argmax(net, 1)
    auc = tf.metrics.auc(label_batch, net_1[:,1])
    print(auc)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    var_dict = adda.util.collect_vars(model)
    restorer = tf.train.Saver(var_list=var_dict)
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    logging.info('Evaluating {}'.format(weights))
    restorer.restore(sess, weights)

    class_correct = np.zeros(dataset.num_classes, dtype=np.int32)
    class_counts = np.zeros(dataset.num_classes, dtype=np.int32)
    for i in tqdm(range(len(split))):
        predictions, gt = sess.run([net, label_batch])
        class_counts[gt[0]] += 1
        if predictions[0] == gt[0]:
            class_correct[gt[0]] += 1
        # print(predictions)
    AUC_val = sess.run(auc[0])
    print("AUC : ", AUC_val)
    logging.info('Class accuracies:')
    logging.info('    ' + format_array(class_correct / class_counts))
    logging.info('Overall accuracy:')
    logging.info('    ' + str(np.sum(class_correct) / np.sum(class_counts)))

    coord.request_stop()
    coord.join(threads)
    sess.close()


@click.command()
@click.argument('dataset')
@click.argument('split')
@click.argument('model')
@click.argument('weights')
@click.option('--gpu', default='0')
def eval_adda(dataset, split, model, weights, gpu):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    # graph = tf.Graph()
    # with graph.as_default():
    #     global_step = graph.get_tensor_by_name('global_step:0')
    dataset_name = dataset
    split_name = split
    dataset = adda.data.get_dataset(dataset_name, shuffle=False)
    split = getattr(dataset, split_name)
    model_fn = adda.models.get_model_fn(model)
    im, label = split.tf_ops(capacity=2)
    im = adda.models.preprocessing(im, model_fn)
    graph = tf.Graph()
    with graph.as_default():
        im_batch, label_batch = tf.train.batch([im, label], batch_size=1)

        # tf.reset_default_graph()

        net, layers = model_fn(im_batch, is_training=False)
        net = tf.argmax(net,1)

        metric_dict = {"Accuracy" : tf.contrib.slim.metrics.streaming_accuracy(net, label_batch)}
        if dataset.num_classes == 2:
            metric_dict["AUC"] = tf.contrib.slim.metrics.streaming_auc(net, label_batch)
        names_to_val, names_to_update = tf.contrib.slim.metrics.aggregate_metric_map(metric_dict)

        num_batches = np.ceil(len(split) / float(32))

        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True
        slim.evaluation.evaluation_loop(
            master="",
            checkpoint_dir=weights,
            logdir=weights,
            num_evals=num_batches,
            eval_op=list(names_to_update.values()),
            session_config=config
        )


if __name__ == '__main__':
    main()
    # eval_adda()