import os
import re
import jieba
import pkuseg
import pyhanlp
import thulac
from settings import SEGFILE_PATTERN


class SegTool():
    '''
    '''
    def __init__(self, seg_tool='pku'):
        self.seg_tool = seg_tool
        self.pk_seg = pkuseg.pkuseg()
        self.thu_seg = thulac.thulac(seg_only=True)

    def seg_jieba(self, sentence):
        return list(jieba.cut(sentence))

    def seg_pkuseg(self, sentence):
        '''
        使用北大分词工具进行分词
        :param sentence:
        :return:
        '''
        return self.pk_seg.cut(sentence)

    def  seg_thulac(self, sentence):
        '''
        使用清华分词工具进行分词
        :param sentence:
        :return:
        '''
        return self.thu_seg.cut(sentence, text=True).split(' ')

    def seg_hanlp(self, sentence):
        '''
        使用Hanlp进行分词
        :param sentence:
        :return:
        '''
        pyhanlp_result = []
        for term in pyhanlp.HanLP.segment(sentence):
            pyhanlp_result.append(term.word)
        return pyhanlp_result

    def __call__(self):
        if self.seg_tool=='pku':
            return self.seg_pkuseg
        if self.seg_tool=='jieba':
            return self.seg_jieba
        if self.seg_tool == 'thu':
            return self.seg_thulac
        if self.seg_tool=='hanlp':
            return self.seg_hanlp



def extract_tokens(doc):
    '''
    etract_tokens for stanfordNLP
    :param doc:
    :return:
    '''
    res = []
    for sent in doc.sentences:
        for wrd in sent.words:
            res.append(wrd.text)
    return(res)


class genSentences(object):
    '''
    基于文件位置来获取逐行的text数据
    '''
    def __init__(self, file_loc):
        self.file = open(file_loc, 'r')

    def __iter__(self):
        return self

    def __next__(self):
        for line in self.file:
            yield line.split()


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if re.search(SEGFILE_PATTERN, fname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield line.split()


