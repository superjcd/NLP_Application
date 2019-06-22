from opencc import OpenCC 
from tqdm import tqdm
from settings import CORPUS_RAW, CORPUS_SIM



def convert_to_sim(source, target):
    opencc = OpenCC('hk2s') # 繁体转简体
    with open(target, 'w') as t:
        with open(source, 'r') as f:
            for line in tqdm(f):
                simple = opencc.convert(line)
                t.write(simple)



if __name__ == '__main__':
    convert_to_sim(CORPUS_RAW, CORPUS_SIM)