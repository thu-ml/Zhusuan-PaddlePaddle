import paddle
import paddle.fluid as fluid

class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.p = self.create_parameter(shape=[1], dtype='float32')

    def forward(self, input):
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

