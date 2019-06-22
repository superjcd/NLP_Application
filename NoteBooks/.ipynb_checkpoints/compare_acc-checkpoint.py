import timeit
import jieba
import thulac
import pkuseg
import thulac
import os


# 导入语句
file_loc = '/Users/jiangchaodi/PycharmProjects/NLPRoadMap/data/test_sentence/'
sentences = []
for file in os.listdir(file_loc):
    if file.startswith('sentence'):
        sentences.append(open(file_loc+file, 'r').read())

def jaccard(set1, set2):
    '''
    计算两个set的杰卡德距离
    :param set1: set类型
    :param set2: set类型
    :return:
    '''
    if not isinstance(set1, set):
        raise TypeError('set1 must be set!')
    if not isinstance(set2, set):
        raise TypeError('set2 must be set!')
    share = set1 & set2
    all = set1 | set2
    jaccard_distance = len(share)/len(all)
    return jaccard_distance


#准确率分比较
def compare_accuracy():
    # load models
    seg = pkuseg.pkuseg()
    thu1 = thulac.thulac(seg_only=True)
    for i,sentence in enumerate(sentences, 1):
        right = set(sentence.split('/'))
        merge = ''.join(sentence.split('/'))
        # 结巴
        target_jieba = set(jieba.cut(merge))
        # pkusg
         # 程序会自动下载所对应的细领域模型
        target_pk = set(seg.cut(merge))
        # 清华
        target_thu = set(thu1.cut(merge, text=True).split(' '))  # 进行一句话分词
        print(f'第{i}句结巴分词的杰卡德距离等于：{jaccard(right, target_jieba)}')
        print(f'第{i}句pkuseg分词的杰卡德距离等于：{jaccard(right, target_pk)}')
        print(f'第{i}句thu分词的杰卡德距离等于：{jaccard(right, target_thu)}')
        print('*'*10)


# sentence = ''.join(sentence1.split('/'))
# pk_seg = pkuseg.pkuseg()
# time_pk = timeit.timeit('pk_seg.cut(sentence)', 'from __main__ import pk_seg, sentence', number=100)
# print(time_pk)



if __name__ == '__main__':
     compare_accuracy()



