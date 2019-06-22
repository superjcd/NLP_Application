import pandas as pd
import requests
import pkuseg
import jieba
import thulac
import pyhanlp
import stanfordnlp
from snownlp import SnowNLP
from datetime import datetime

requests.DEFAULT_RETRIES = 8

class SegToolsMetaclass(type):
    def __new__(cls, name, bases, attrs):
        count = 0
        attrs['__segtools__'] = []
        for k, v in attrs.items():
            if 'seg_' in k:
                attrs['__segtools__'].append(k)
        return type.__new__(cls, name, bases, attrs)



class SegTools(object, metaclass=SegToolsMetaclass):
    '''
    '''
    def __init__(self):
        self.tools = []
        self._n = 0
        self.pk_seg = pkuseg.pkuseg()
        self.thu_seg = thulac.thulac(seg_only=True)
        self.stanford_pipeline = stanfordnlp.Pipeline(processors='tokenize', lang='zh')


    # def seg_jieba(self, sentence):
    #     return list(jieba.cut(sentence))
    #
    # def seg_pkuseg(self, sentence):
    #     '''
    #     使用北大分词工具进行分词
    #     :param sentence:
    #     :return:
    #     '''
    #     return self.pk_seg.cut(sentence)
    #
    # def  seg_thulac(self, sentence):
    #     '''
    #     使用清华分词工具进行分词
    #     :param sentence:
    #     :return:
    #     '''
    #     return self.thu_seg.cut(sentence, text=True).split(' ')
    #
    # def seg_hanlp(self, sentence):
    #     '''
    #     使用Hanlp进行分词
    #     :param sentence:
    #     :return:
    #     '''
    #     pyhanlp_result = []
    #     for term in pyhanlp.HanLP.segment(sentence):
    #         pyhanlp_result.append(term.word)
    #     return pyhanlp_result
    #
    #
    # def seg_snownlp(self, sentence):
    #     '''
    #     使用snownlp进行分词
    #     :param sentence:
    #     :return:
    #     '''
    #     return SnowNLP(sentence).words
    #
    # def seg_stanfordnlp(self, sentence):
    #     try:
    #         doc = self.stanford_pipeline(sentence)
    #         result = extract_tokens_pystf(doc)
    #     except:
    #         result=[]
    #     return result

    def seg_stanfordcorenlp(self, sentence):
        import requests
        params = (
            ('properties', '{"annotators":"tokenize","outputFormat":"json"}'),
        )

        data = sentence.encode('utf-8')
        response = requests.post('http://localhost:9100/', params=params, data=data)
        if response.status_code == 503:
            print(response.status_code)
            raise Exception(
                '请确定在指定目录下已经启动了server, 请参阅：https://stanfordnlp.github.io/stanfordnlp/installation_download.html')
        elif response.status_code == 500:
            print('输入的是空字符串')
            return ([])
        else:
            res = response.json()
        return extract_tokens_stfcore(res)



def extract_tokens_pystf(doc):
    '''
    etract_tokens for stanfordNLP
    :param doc:
    :return:
    '''
    res = []
    for sent in doc.sentences:
        for wrd in sent.words:
            res.append(wrd.text)
    return res


def extract_tokens_stfcore(doc):
    res = []
    for tok in doc['tokens']:
        res.append(tok['word'])
    return res


def cal_acc(result_list, save=True):
    print(f'有{len(result_list)}个分词工具需要计算结果:')
    res = []
    for result in result_list:
        temp = {}
        temp['NLP工具'] = result.__name__.split('_')[-1]
        temp['准确率'] = result.acc_words_len/result.can_words_len
        temp['召回率'] = result.acc_words_len/result.ref_words_len
        temp['F1分数'] = (2 *temp['准确率']*temp['召回率'])/(temp['准确率']+temp['召回率'])
        print(f"{temp['NLP工具']}的结果是：{temp}")
        res.append(temp)
    if save:
        data = pd.DataFrame(res)
        file_name = 'result/acc/' + format(datetime.now(), '%m%d-%H%M%S') + '分词准确率对比.csv'
        data.to_csv(file_name, index=False, encoding='utf_8_sig')


def cal_time(result, save=True):
    print(f'有{len(result)}个分词工具需要计算结果:')
    res = []
    data = pd.DataFrame(result)
    print(f'我们对运行时间对评估结果是：{data}')
    if save:
        file_name = 'result/time/' + format(datetime.now(), '%m%d-%H%M%S') + '分词速度对比.csv'
        data.to_csv(file_name, index=False, encoding='utf_8_sig')



def compare_line(reference, candidate): # reference 标注
    '''

    :param reference: 正确分词语句
    :param candidate: 比较对象
    :return:
    '''
    ref_words = reference.split()
    can_words = candidate.split()

    ref_words_len = len(ref_words)
    can_words_len = len(can_words)

    ref_index = words_to_index(ref_words)
    can_index = words_to_index(can_words)

    tmp = [val for val in ref_index if val in can_index]
    acc_word_len = len(tmp)

    return ref_words_len, can_words_len, acc_word_len


def words_to_index(seg_sentence):
    index = 0
    res = []
    for word in seg_sentence:
        word_index = [index]
        index += len(word)
        word_index.append(index)
        res.append(word_index)
    return res



def seg_stanfordcore(sentence):
    import requests
    params = (
        ('properties', '{"annotators":"tokenize","outputFormat":"json"}'),
    )

    data = sentence.encode('utf-8')
    response = requests.post('http://localhost:9100/', params=params, data=data)
    if response.status_code == 503:
        print(response.status_code)
        raise Exception(
            '请确定在指定目录下已经启动了server, 请参阅：https://stanfordnlp.github.io/stanfordnlp/installation_download.html')
    elif response.status_code == 500:
        print('输入的是空字符串')
        return([])
    else:
        res = response.json()
    print(extract_tokens_stfcore(res))


if __name__ == '__main__':
    seg_stanfordcore('')
    # sentence = ' 我是你爸爸'
    # st = SegTools()
    # for attributes in dir(st):
    #     if attributes.startswith('seg'):
    #         print(attributes)
    #         print(getattr(st, attributes)(sentence))


