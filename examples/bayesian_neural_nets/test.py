import paddle
import paddle.fluid as fluid
import numpy as np

class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.p = self.create_parameter(shape=[1], dtype='float32')

    def forward(self, input):
        t1 = paddle.to_tensor(np.log(np.random.rand(*[1000,10])), dtype='float32')
        t2 = paddle.to_tensor(np.log(np.random.rand(*[1000,10])), dtype='float32')
        for i in range(10000):
            print(i)
            event = paddle.to_tensor(np.log(np.random.rand(1000,10)), dtype='float32')
            a = paddle.cast(t2>event, dtype='float32')
            t1 = paddle.assign(a * t2 + (1.0 - a) * t1)
        #return paddle.tile(self.p, [2, 1])
        return paddle.concat([self.p, self.p], 0)


x = paddle.randn([10, 1], 'float32')
mylayer = MyLayer()
mylayer.train()

opt = paddle.optimizer.Adam(learning_rate=0.001,
			    parameters=mylayer.parameters(),)

out = mylayer(x)
print(out)
out.backward()
print('p.grad: ', mylayer.p.grad)

opt.step()

