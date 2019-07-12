"""
##############################
# LSTM Network configuration #
##############################
"""


class config():
    size = 256
    n_percept_hidden_layer = 4
    n_percept_filters = 32
    beta_start = .01
    beta_end = .001
    decay_steps = 50000000
    summary_steps = 10
