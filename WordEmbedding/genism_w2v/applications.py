from gensim.models import Word2Vec


class W2VApp():
    def __init__(self, model_loc):
        self.model = Word2Vec.load(model_loc)

    def xsd(self, A, B):
        '''
        计算A和B的相似度
        :param A: 比较词语A
        :param B: 比较词语B
        :return: 相似度， 越接近1， 相似度越高
        '''
        return self.model.wv.similarity(A, B)

    def most_similar(self, A, topn=3):
        return self.model.wv.most_similar(A, topn=topn)

    def analogy(self, A, B, C, n=1):
        '''
        基于A、B、C， 输出D， 基于类比关系A->B:C-D
        :param A:
        :param B:
        :param C:
        :param n:
        :return:
        '''
        return self.model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=n)


    def get_vector(self, word):
        '''
        获取词语的numpy array形式
        :param word:  中文单词
        :return: numpy array
        '''
        return self.model.wv[word]

    def not_match(self, words):
        return self.model.wv.doesnt_match(words)


if __name__ == '__main__':
    app = W2VApp(model_loc='model/zh_wiki.model')
    vector = app.get_vector('我')
    print(vector.shape)
    # print(app.most_similar('国王'))
    # print(app.xsd('足球', '男足'))
    # print(app.analogy('男人', '女人', '国王'))
    # print(app.get_vector('疯子'))
    # print(app.not_match(['男人', '女人', '房子', '麦片']))









