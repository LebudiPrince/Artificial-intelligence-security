在实验过程中发现关键的参数是学习率和batch size
在优化器为普通的梯度下降时，学习率为0.01时，模型的准确率只有20%左右，当调整为0.001，准确率就达到了56%。
batch size不宜设置得过大，会加大每次计算的资源，并且并不能得到更好的效果。
此外我们也可以更改优化器，改成Adam会有更好的效果，准确率可以提升到63%。