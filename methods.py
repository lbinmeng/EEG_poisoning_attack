import tensorflow as tf
import numpy as np
import random
from lib import utils


class CWL2(object):
    def __init__(self, sess, model, initial_c, batch_size, learning_rate, target, binary_search_step, input_shape,
                 max_iteration):
        self.sess = sess
        self.model = model
        self.initial_c = initial_c
        self.batch_size = batch_size
        self.lr = learning_rate
        self.target = target
        self.binary_search_step = binary_search_step
        self.shape = [batch_size, input_shape[0], input_shape[1]]
        self.max_iteration = max_iteration

        # variable to optimize
        modifier = tf.Variable(np.zeros(shape=self.shape), dtype=tf.float32)

        self.tdata = tf.Variable(np.zeros(self.shape), dtype=tf.float32, name='tdata')
        self.const = tf.Variable(np.zeros(self.batch_size), dtype=tf.float32, name='const')

        self.assign_tdata = tf.placeholder(tf.float32, shape=self.shape, name='assign_tdata')
        self.assign_const = tf.placeholder(tf.float32, shape=[batch_size], name='assign_const')

        # clip the example
        self.newdata = (tf.tanh(modifier + self.tdata) + 1) / 2
        self.oridata = (tf.tanh(self.tdata) + 1) / 2

        # prediction of model
        self.toutput = model(self.oridata)
        self.output = model(self.newdata)

        # L2 norm
        self.l2 = tf.reduce_sum(tf.square(self.newdata - self.oridata), axis=[1, 2])

        # Loss
        loss_pre = tf.maximum(self.toutput + self.target - self.output, np.asarray(0., dtype=np.float32))
        self.loss1 = tf.reduce_sum(self.const * loss_pre)
        self.loss2 = tf.reduce_sum(self.l2)
        self.loss = self.loss1 + self.loss2

        # Train variables
        start_vars = set(x.name for x in tf.global_variables())
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # Initial
        self.setup = []
        self.setup.append(self.tdata.assign(self.assign_tdata))
        self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, data):
        batch_size = self.batch_size
        l2 = []
        [adv, l2_temp] = self.attack_batch(data[:batch_size])
        l2.append(l2_temp)
        for i in range(1, len(data) // batch_size):
            print('Running CW attack on instance {} of {}'.format(i * batch_size, len(data)))
            # adv.append(self.attck_batch(data[i * batch_size:(i + 1) * batch_size]))
            [temp_adv, l2_temp] = self.attack_batch(data[i * batch_size:(i + 1) * batch_size])
            adv = np.concatenate([adv, temp_adv], axis=0)
            l2.append(l2_temp)

        if len(data) % batch_size != 0:
            last_idx = len(data) - (len(data) % batch_size)
            print('Running CW attack on instace {} of {}'.format(last_idx, len(data)))
            temp_data = np.zeros((batch_size,) + data.shape[1:])
            temp_data[:(len(data) % batch_size)] = data[last_idx:]
            [temp_adv, l2_temp] = self.attack_batch(temp_data)
            adv = np.concatenate([adv, temp_adv[:(len(data) % batch_size)]], axis=0)
            l2.append(l2_temp)
        return np.array(adv), np.mean(np.array(l2))

    def attack_batch(self, data):
        batch_size = self.batch_size

        data = np.clip(data, 0, 1)
        # convert to tanh-space
        data = (data * 2) - 1
        data = np.arctanh(data)

        lower_bound = np.zeros(batch_size)
        const = np.ones(batch_size) * self.initial_c
        upper_bound = np.ones(batch_size) * 1e10

        # best l2, score, instance
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = np.copy(data)

        l2 = 0
        for step in range(self.binary_search_step):
            self.sess.run(self.init)
            batch = data[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            print('Binary search step {} of {}'.format(step, self.binary_search_step))

            self.sess.run(self.setup, {self.assign_tdata: batch, self.assign_const: const})

            for iteration in range(self.max_iteration):
                _, l, l2, pred_adv, pred, ndata, c = self.sess.run(
                    [self.train_step, self.loss, self.l2, self.output, self.toutput, self.newdata, self.const])

                if iteration % 50 == 0:
                    print(
                        'Iteration {} of {}: loss={:.3g} l2={:.3g} pred_adv={:.3g} pred={:.3g} c={:.3g}'.format(
                            iteration,
                            self.max_iteration, l,
                            np.mean(l2),
                            np.mean(pred_adv),
                            np.mean(pred),
                            np.mean(c)))

                # adjust the best result found so far
                for e, (dst, pre_a, pre, nd) in enumerate(zip(l2, pred_adv, pred, ndata)):
                    if dst < bestl2[e] and pred_adv[e] >= pred[e] + self.target:
                        bestl2[e] = dst
                        bestscore[e] = pre_a
                    if dst < o_bestl2[e] and pred_adv[e] >= pred[e] + self.target:
                        o_bestl2[e] = dst
                        o_bestscore[e] = pre_a
                        o_bestattack[e] = nd

            # adjust the constant
            for e in range(batch_size):
                if pred_adv[e] >= pred[e] + self.target and bestscore[e] != -1:
                    upper_bound[e] = min(upper_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    lower_bound[e] = max(lower_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        const[e] *= 10

            print('Sucessfully generated adversarial examples on {} of {} instance.'.format(sum(upper_bound < 1e9),
                                                                                            batch_size))
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(o_bestl2[o_bestl2 < 1e9])
            l2 = mean
            print('Mean successful distortion: {:.4g}'.format(mean))

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack, l2


def mask_opt(sess, model, inputs, labels, lr=0.001, alpha=0.001, iterations=100):
    """ get attack mask """
    input_shape = [inputs.shape[1], inputs.shape[2], inputs.shape[3]]
    mask = tf.Variable(tf.zeros(shape=[*input_shape], dtype=tf.float32), name='mask')
    inputs_place = tf.placeholder(tf.float32, shape=[None, *input_shape])
    labels_place = tf.placeholder(tf.int32, shape=[None])

    # mask_ = (tf.sign(mask) + 1) / 2
    # adv = tf.multiply(mask_, inputs_place)
    adv = mask + inputs_place
    # prediction of adversarial examples
    pred = model(adv)
    loss1 = tf.losses.sparse_softmax_cross_entropy(labels_place, pred)
    loss2 = tf.reduce_sum(tf.abs(mask))
    # loss2 = tf.reduce_sum(tf.square(mask))
    loss = - loss1 + alpha * loss2

    start_vars = set(x.name for x in tf.global_variables())
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=[mask])
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]

    sess.run(tf.variables_initializer(var_list=[mask] + new_vars))

    for iter in range(iterations):
        _, l, l1, l2, adv_mask, = sess.run([train_step, loss, loss1, loss2, mask],
                                           feed_dict={inputs_place: inputs, labels_place: labels})

        print('Iter:{}, loss:{}, loss1:{}, loss2:{}'.format(iter, l, l1, l2))

    return adv_mask


def iterative_opt(sess, model, inputs, labels, iterations=50):
    """ iterative optimize mask """
    input_shape = [inputs.shape[1], inputs.shape[2], inputs.shape[3]]
    mask_place = tf.placeholder(tf.float32, shape=input_shape)
    inputs_place = tf.placeholder(tf.float32, shape=[None, *input_shape])
    labels_place = tf.placeholder(tf.int32, shape=[None])

    adv = tf.multiply(mask_place, inputs_place)
    # prediction of adversarial examples
    pred = model(adv)
    loss = tf.losses.sparse_softmax_cross_entropy(labels_place, pred)

    mask = np.ones(shape=input_shape)
    for iter in range(iterations):
        best_position = 0
        max_loss = 0.0
        for i in range(input_shape[2]):
            mask_ = mask.copy()
            mask_[:, :, i] = 0.0
            l = sess.run(loss, feed_dict={inputs_place: inputs, labels_place: labels, mask_place: mask_})
            if l > max_loss:
                max_loss = l
                best_position = i
        mask[:, :, best_position] = 0.0

    return mask


def mask_search(mask, mask_len):
    """ search the mask with largest value"""
    mask_position = 0
    mask_sum = 0.0

    # search mask start position
    for i in range(0, mask.shape[2] - mask_len):
        sum = np.sum(np.abs(mask[:, :, i:i + mask_len]))
        if sum > mask_sum:
            mask_sum = sum
            mask_position = i

    mask[:, :, :mask_position], mask[:, :, mask_position + mask_len:] = 0.0, 0.0

    return mask


def loss_mask(mask, mask_len, mask_num):
    """ packet loss position """
    index = [i for i in range(mask.shape[2] - mask_len)]
    mask_ = np.ones(shape=mask.shape)

    # search mask start position
    for num in range(mask_num):
        mask_position = 0
        mask_sum = 0.0
        for i in index:
            sum = np.sum(np.abs(mask[:, :, i:i + mask_len]))
            if sum > mask_sum:
                mask_sum = sum
                mask_position = i

        mask_[:, :, mask_position:mask_position + mask_len] = 0.0
        # remove the position already chosen
        for r in range(2 * mask_len):
            if mask_position - mask_len + r in index:
                index.remove(mask_position - mask_len + r)

    return mask_


def random_mask(shape, mask_len, mask_num):
    """" random mask """
    index = [i for i in range(shape[2] - mask_len)]
    mask = np.ones(shape=shape)

    for num in range(mask_num):
        mask_position = random.sample(index, 1)
        mask_position = mask_position[0]
        mask[:, :, mask_position:mask_position + mask_len] = 0.0
        for r in range(mask_len):
            if mask_position + r - 1 in index:
                index.remove(mask_position + r - 1)

    return mask


def pulse_noise(shape, freq, sample_freq, proportion, phase=.0):
    """ generate pulse noise """
    length = shape[2]
    proportion = 1 - proportion
    t = 1 / freq
    k = int(length / (t * sample_freq))
    pulse = np.zeros(shape=shape)

    for i in range(k):
        pulse[:, :, int((i + proportion - phase) * t * sample_freq):int((i + 1 - phase) * t * sample_freq)] = 1.0

    if length > int((i + 1 + proportion - phase) * t * sample_freq):
        pulse[:, :, int((i + 1 + proportion - phase) * t * sample_freq):] = 1.0

    return pulse
