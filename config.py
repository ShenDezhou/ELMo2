
class Config(object):
    def __init__(self):
        self.max_tok_len = 20     #单词最大长度
        self.train_file = './data_processed/product_comment_processed.txt'
        self.dev_file = './data_processed/product_comment_processed_dev.txt'
        self.char_embedding_dim = 16
        self.char_conv_filters = [[1,32], [2,32]]
        self.num_highways = 0 #Highway网络层数
        self.projection_dim = 32  #投射维度，从lstmhidden投射
        self.hidden_dim = 32   #lstm隐含层维度
        self.num_layers = 1     #lstm层数
        self.batch_size = 16
        self.dropout = 0.1
        self.learning_rate = 0.0004
        self.clip_grad = 5   #梯度最大范数，用于训练梯度裁剪
        self.num_epoch = 10
        self.save_path = './trained_model/elmo_bilm'
        self.use_cuda = False



    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':

    con = Config()
    print(con)