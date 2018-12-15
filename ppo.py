from keras.models import Model, model_from_json, load_model
from keras.optimizers import Adam, RMSprop
import os
from keras.layers import Input, Dense, Conv2D, Flatten,BatchNormalization, Activation, Dropout, MaxPooling2D
from keras.layers.merge import concatenate
import keras.backend as K
import time
from copy import deepcopy
import numpy as np


class Memory:
    def __init__(self):
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_s_ = []
        self.batch_done = []

    def store(self, s, a, s_, r, done):
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)

    def clear(self):
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()

    @property
    def cnt_samples(self):
        return len(self.batch_s)


def conv2d_bn(input_layer, index_layer,
              filters=16,
              kernel_size=(3, 3),
              strides=(1, 1)):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  name="conv{0}".format(index_layer))(input_layer)
    bn = BatchNormalization(axis=bn_axis, scale=False, name="bn{0}".format(index_layer))(conv)
    act = Activation('relu', name="act{0}".format(index_layer))(bn)
    pooling = MaxPooling2D(pool_size=2)(act)
    # x = Dropout(0.3)(pooling)
    return pooling


class Agent:
    def __init__(self, dic_agent_conf, dic_path, dic_env_conf):
        self.dic_agent_conf = dic_agent_conf
        self.dic_path = dic_path
        self.dic_env_conf = dic_env_conf

        self.n_actions = self.dic_env_conf["ACTION_DIM"]

        self.actor_network = self._build_actor_network()
        self.actor_old_network = deepcopy(self.actor_network)

        self.critic_network = self._build_critic_network()

        self.memory = Memory()

    def choose_action(self, state):
        assert isinstance(state, dict), "state must be dictionary"
        state = self.convert_state_to_input(state)

        prob = self.actor_network.predict_on_batch(state)
        action = np.random.choice(self.n_actions, prob)
        return action

    def train_network(self):
        n = self.memory.cnt_samples
        discounted_r = []
        if self.memory.batch_done[-1]:
            v = 0
        else:
            v = self.get_v(self.memory.batch_s_[-1])
        for r in self.memory.batch_r[::-1]:
            v = r + self.dic_agent_conf["GAMMA"] * v
            discounted_r.append(v)
        discounted_r.reverse()

        batch_s, batch_a, batch_discounted_r = np.vstack(self.memory.batch_s), \
                     np.vstack(self.memory.batch_a), \
                     np.vstack(discounted_r)[:, np.newaxis]

        self.update_target_network()

        batch_v = self.critic_network.predict(batch_s)
        batch_advantage = batch_discounted_r - batch_v
        batch_old_prediction = self.actor_old_network.predict(batch_s)

        batch_a = np.vstack([np.array(np.arange(n), batch_a)])
        self.actor_network.train_on_batch(x=[batch_s, batch_advantage, batch_old_prediction], y=batch_a)
        self.critic_network.train_on_batch(x=batch_s, y=batch_discounted_r)
        self.memory.clear()

    def store_transition(self, s, a, s_, r, done):
        self.memory.store(s, a, s_, r, done)

    def get_v(self, s):
        v = self.actor_network.predict_on_batch(s)
        return v

    def save_model(self, file_name):
        self.actor_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_actor_network.h5" % file_name))
        self.critic_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_network.h5" % file_name))

    def load_model(self):
        self.actor_network = load_model(self.dic_path["PATH_TO_MODEL"], "%s_actor_network.h5" % file_name)
        self.critic_network = load_model(self.dic_path["PATH_TO_MODEL"], "%s_critic_network.h5" % file_name)
        self.actor_old_network = deepcopy(self.actor_network)

    def convert_state_to_input(self, state):
        return [state[feature_name]
                for feature_name in self.dic_env_conf["LIST_STATE_FEATURE"]]

    def _build_actor_network(self):
        dic_input_node = {}
        for feature_name in self.dic_env_conf["LIST_STATE_FEATURE"]:
            dic_input_node[feature_name] = Input(self.dic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()],
                                                 name="input_"+feature_name)
        advantage = Input(shape=1, name="Advantage")
        old_prediction = Input(shape=self.n_actions, name="Old_Prediction")

        # add cnn to image feature
        dic_flatten_node = {}
        for feature_name in self.dic_env_conf["LIST_STATE_FEATURE"]:
            if len(self.dic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]) > 1:
                dic_flatten_node[feature_name] = self._cnn_network_structure(dic_input_node[feature_name])
            else:
                dic_flatten_node[feature_name] = dic_input_node[feature_name]

        list_all_flatten_feature = []
        for feature_name in self.dic_env_conf["LIST_STATE_FEATURE"]:
            list_all_flatten_feature.append(dic_flatten_node[feature_name])
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")

        shared_hidden = self._shared_network_structure(all_flatten_feature)

        action_dim = self.dic_env_conf["ACTION_DIM"]

        policy = Dense(action_dim, activation="softmax", name="actor_output_layer")(shared_hidden)

        actor_network = Model(inputs=list(dic_input_node.values()), outputs=policy)

        if self.dic_agent_conf["OPTIMIZER"] is "Adam":
            actor_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                  loss=self.proximal_policy_optimization_loss(
                                    advantage=advantage, old_prediction=old_prediction,
                                  ))
        elif self.dic_agent_conf["OPTIMIZER"] is "RMSProp":
            actor_network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]))
        else:
            print("Not such optimizer for actor network. Instead, we use adam optimizer")
            actor_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]))
        actor_network.summary()

        time.sleep(0.5)
        return actor_network

    def update_target_network(self):
        alpha = self.dic_agent_conf["TARGET_UPDATE_ALPHA"]
        self.actor_old_network.set_weights(alpha*self.actor_network.get_weights()
                                           + (1-alpha)*self.actor_old_network.get_weights())

    def _build_critic_network(self):
        dic_input_node = {}
        for feature_name in self.dic_env_conf["LIST_STATE_FEATURE"]:
            dic_input_node[feature_name] = Input(self.dic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()],
                                                 name="input_"+feature_name)
        # add cnn to image feature
        dic_flatten_node = {}
        for feature_name in self.dic_env_conf["LIST_STATE_FEATURE"]:
            if len(self.dic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]) > 1:
                dic_flatten_node[feature_name] = self._cnn_network_structure(dic_input_node[feature_name])
            else:
                dic_flatten_node[feature_name] = dic_input_node[feature_name]

        list_all_flatten_feature = []
        for feature_name in self.dic_env_conf["LIST_STATE_FEATURE"]:
            list_all_flatten_feature.append(dic_flatten_node[feature_name])
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")

        shared_hidden = self._shared_network_structure(all_flatten_feature)

        action_dim = self.dic_env_conf["ACTION_DIM"]

        if self.dic_env_conf["POSITIVE_REWARD"]:
            q = Dense(action_dim, activation="relu", name="critic_output_layer")(shared_hidden)
        else:
            q = Dense(action_dim, name="critic_output_layer")(shared_hidden)

        critic_network = Model(inputs=list(dic_input_node.values()), outputs=q)

        if self.dic_agent_conf["OPTIMIZER"] is "Adam":
            critic_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                   loss=self.dic_agent_conf["CRITIC_LOSS"])
        elif self.dic_agent_conf["OPTIMIZER"] is "RMSProp":
            critic_network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                   loss=self.dic_agent_conf["CRITIC_LOSS"])
        else:
            print("Not such optimizer for actor network. Instead, we use adam optimizer")
            critic_network.compile(optimizer=Adam(lr=self.dic_agent_conf["ACTOR_LEARNING_RATE"]),
                                   loss=self.dic_agent_conf["CRITIC_LOSS"])
        critic_network.summary()

        time.sleep(0.5)
        return critic_network

    @staticmethod
    def _cnn_network_structure(img_features):
        conv1 = conv2d_bn(img_features, 1, filters=32, kernel_size=(8, 8), strides=(4, 4))
        conv2 = conv2d_bn(conv1, 2, filters=16, kernel_size=(4, 4), strides=(2, 2))
        img_flatten = Flatten()(conv2)
        return img_flatten

    def _shared_network_structure(self, state_features):
        dense_d = self.dic_agent_conf["D_DENSE"]
        hidden1 = Dense(dense_d, activation="relu", name="hidden_shared_1")(state_features)
        hidden2 = Dense(dense_d, activation="relu", name="hidden_shared_2")(hidden1)
        return hidden2

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        loss_clipping = self.dic_agent_conf["LOSS_CLIPPING"]
        entropy_loss = self.dic_agent_conf["ENTROPY_LOSS"]

        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping,
                                                           max_value=1 + loss_clipping) * advantage) + entropy_loss * (
                           prob * K.log(prob + 1e-10)))

        return loss
