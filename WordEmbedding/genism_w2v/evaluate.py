from gensim.models import Word2Vec


if __name__ == '__main__':
    model =  Word2Vec.load('model/zh_wiki.model')
    # words = model.most_similar(u"足球")
    # for t in words:
    #     print(t[0],t[1])
    print(model.similarity(u'书籍',u'书本'))
    word = model.most_similar(positive=[u'皇上',u'国王'],negative=[u'皇后'])
    for t in word:
        print(t[0],t[1])
    print(model.doesnt_match(u'太后 妃子 贵人 贵妃 才人'.split()))
    print(model.similarity(u'书籍',u'书本'))
    print(model.similarity(u'逛街',u'书本'))