import paddle

print(paddle.log(paddle.to_tensor([1e-8], dtype='float32')))
