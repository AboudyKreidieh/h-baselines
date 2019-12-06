import tensorflow as tf
import multiprocessing
import os


def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.compat.v1.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess


def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.compat.v1.InteractiveSession(config=config, graph=graph)
    else:
        return tf.compat.v1.Session(config=config, graph=graph)
