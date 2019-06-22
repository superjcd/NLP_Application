from tqdm import tqdm
from utilities import SegTool
from settings import CORPUS_SIM, CORPUS_SEG, WORD_SEG_TOOL


def segment(source, target, tool=WORD_SEG_TOOL):
    seg_tool = SegTool(seg_tool=tool)()
    with open(target, 'w') as t:
        with open(source, 'r') as f:
            for line in tqdm(f):
                segmented = ' '.join(seg_tool(line))
                t.write(segmented)


if __name__ == '__main__':
    segment(CORPUS_SIM, CORPUS_SEG)
