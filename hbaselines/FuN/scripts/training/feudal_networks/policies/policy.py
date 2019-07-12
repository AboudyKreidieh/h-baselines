"""
##########################################################
# Script containing the base Neural Network policy class #
##########################################################
"""


class Policy(object):
    """
    An abstract class defining a learned policy to be used for a Reinforcment
    Learning problem.  This class interfaces with a policy optimizer class
    that oversees training the policy on some environment.

    The policy needs three externally facing methods:
        act()
        value()
        update()
    Which are further documented below.

    Further, upon initialization the following member variables should be
    defined:
        loss        - The tensorflow operation defining the
                        loss function of the
                      policy with respect to a batch of training data
        var_list    - The variables that should be trained by the optimizer
        internals_in- A list of placeholder variables needed at runtime
                      in order to calculate act(), value() or update()
                      (e.g. internal LSTM state)
    """
    def __init__(self, obs_space, act_space, config):
        """
         Instantiate an Neural Network policy object.

        Parameters
        ----------
        obs_space : object
            Observation space
        act_space : object
            Action space
        config : object
            Configuration of the neural network
        """
        raise NotImplementedError("Please Implement this method")

    def _build_model(self):
        """
        Private utility function that builds models.

        """
        raise NotImplementedError("Please Implement this method")

    def _build_placeholders(self):
        """
        Private utility function that helps build the network placeholders

        """
        raise NotImplementedError("Please Implement this method")

    def _build_loss(self):
        """
        Private utility function that builds losses for models.

        """
        raise NotImplementedError("Please Implement this method")

    def act(self, obs, prev_internal):
        """
        Function to allow the network to start acting
        based on the environmental
        observations and previous internal states.

        Parameters
        ----------
        obs : object
            Observation object
        prev_internal : object
            Previous internal states of the neural network
        """
        raise NotImplementedError("Please Implement this method")

    def value(self, obs, prev_internal):
        """
        Value function for the network

        Parameters
        ----------
        obs : object
            Observation object
        prev_internal : object
            Previous internal states of the neural network
        """
        raise NotImplementedError("Please Implement this method")

    def update(self, sess, train_op, batch):
        """
        Update function for the network

        Parameters
        ----------
        sess : object
            Session object
        train_op : object
            BLANK
        batch : object
            Batch object of data for the neural network
        """
        raise NotImplementedError("Please Implement this method")
