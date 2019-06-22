import time
import fire
from collections import namedtuple
from tqdm import tqdm
from utility import SegTools, compare_line, cal_time, cal_acc


class EvaluateAcc():
    '''
    评估准确度
    '''
    def __init__(self, segment_file, raw_file):
        self.segment_file = segment_file
        self.raw_file = raw_file
        self.resultcollecter = []

    def main(self):
        st = SegTools()
        for seg_tool in st.__segtools__:
            segment_file = open(self.segment_file, 'r')
            raw_file = open(self.raw_file, 'r')
            result = namedtuple(seg_tool, ['ref_words_len', 'can_words_len', 'acc_words_len'])
            result.ref_words_len = 0
            result.can_words_len = 0
            result.acc_words_len = 0
            print(f"开始评估分词工具{seg_tool.split('_')[-1]}准确率")
            for segment, raw in tqdm(zip(segment_file, raw_file)):
                target = getattr(st, seg_tool)(raw)
                ref_words_len, can_words_len, acc_words_len = compare_line(segment, ' '.join(target))
                result.ref_words_len += ref_words_len
                result.can_words_len += can_words_len
                result.acc_words_len += acc_words_len
            print(result.ref_words_len, result.can_words_len, result.acc_words_len)
            self.resultcollecter.append(result)


class EvaluateTime():
    '''
    评估速度
    '''
    def __init__(self, raw_file, lines=10000, file_name='MSR'):
        self.raw_file = raw_file
        self._lines = lines
        self.resultcollecter = []
        self.file_name = file_name

    def main(self):
        st = SegTools()
        for seg_tool in st.__segtools__:
            n = 0
            raw_file = open(self.raw_file, 'r')
            result = {}
            result['数据集'] = self.file_name
            result['语句数量'] = self._lines
            result['分词工具'] = seg_tool.split('_')[-1]
            start = time.time()
            print(f"开始使用分词工具{result['分词工具']}")
            for raw in raw_file:
                if n <= self._lines:
                    # run the tool
                    getattr(st, seg_tool)(raw)
                    n += 1
            end = time.time()
            result['运行时间'] = end - start
            self.resultcollecter.append(result)


def eval_acc(segment_file='data/msr_segmented.txt', raw='data/msr_raw.txt'):
    eva_acc = EvaluateAcc(segment_file, raw)
    eva_acc.main()
    result = eva_acc.resultcollecter
    cal_acc(result)


def eval_time(raw='data/msr_raw.txt', n=10000):
    eva_time = EvaluateTime(raw, lines=n)
    eva_time.main()
    result = eva_time.resultcollecter
    cal_time(result)



if __name__=='__main__':
    fire.Fire()