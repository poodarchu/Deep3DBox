import os
import mxnet as mx
import numpy as np

"""
    This script is used to try out MXNet features.
"""

class Softmax(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.exp(x-x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()
        y[np.arange(l.shape[0]), 1] -= 1.0
        self.assign(in_grad[0], req[0], mx.nd.array(y))

    # def assign(self, dst, req, src):
    #     pass


# Softmax defines the computation of our custom operator, but you still need to define
# its input/output format by subclassing mx.operator.CustomProp.
# First, register the new operator with the name 'softmax'
@mx.operator.register('softmax')
class SoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SoftmaxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Softmax()

fc3 = mx.sym.FullyConnected()
mlp = mx.symbol.Custom(data=fc3, name='softmax_custom', op_type='softmax')

mx.random.seed(1234)

fname = mx.test_utils.download('http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data')
data = np.genfromtxt(fname, delimiter=',')[:, 1:]
label = np.array([ord(l.split(',')[0])-ord('A') for l in open(fname, 'r')])

batch_size = 32
ntrain = int(data.shape[0]*0.8)
train_iter = mx.io.NDArrayIter(data[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data[ntrain:, :], label[ntrain:], batch_size)

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type='relu')
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=26)
net = mx.sym.SoftmaxOutput(net, name='softmax')
mx.viz.plot_network(net)

# Intermediate Level Interface.
mod = mx.module.Module(symbol=net, context=mx.gpu(), data_names=['data'], label_names=['softmax_label'])
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
mod.init_params(initializer=mx.init.Uniform(scale=.1))
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
metric = mx.metric.create('acc')
for epoch in range(5):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)
        mod.update_metric(metric, batch.label)
        mod.backward()
        mod.update()
    print('Epoch %d, Training %s' % (epoch, metric.get()))


# High-level Interface
train_iter.reset()
mod = mx.mod.Module(
    symbol=net,
    data_names=['data'],
    label_names=['softmax_label']
)

# Fit the module
mod.fit(
    train_iter,
    eval_data=val_iter,
    optimizer='sgd',
    optimizer_params={'learning_rate':0.1},
    eval_metric='acc',
    num_epoch=8
)

# Predict and Evaluate
y = mod.predict(val_iter)
assert y.shape == (4000, 26)
# If we don't need the prediction outputs, but just need to evaluate on a test set,
# we can call the score() function. It runs prediction in the input validation dataset.
# and evaluate the performance according to the given input metric
score = mod.score(val_iter, ['acc'])
print(score[0][1])

# Save and Load
model_prefix = 'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix)

mod = mx.mod.Module(symbol=net)
mod.fit(train_iter, num_epoch=5, epoch_end_callback=checkpoint)

sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
mod.set_params(arg_params, aux_params)
# If we just want to resume training from a saved checkpoint, instead of calling set_params(), we can directly call fit()
# Passing the loaded parameters, so that fit() knows to start from those parameters instead of initializing randomly from scratch.
# we also set the begin epoch parameters so that fit() knows we're resuming from a previously saved epoch
mod = mx.mod.Module(symbol=sym)
mod.fit(
    train_iter,
    num_epoch=21,
    arg_params=arg_params,
    aux_params=aux_params,
    begin_epoch=3
)
