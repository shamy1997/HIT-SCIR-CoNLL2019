from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from tqdm import trange
import spacy_udpipe


import collections
import json

nlp = spacy_udpipe.load('en')
# tokenizer = Tokenizer(nlp.vocab)


def load_json(size,js_path,nlp,output):
    mrps = []

    with open(js_path) as fi :
        with open(output,'w') as fo:
            line = fi.readline().strip()
            while line:
                mrp = json.loads(line)
                if 'input' in mrp:
                    tokens = [t.text for t in nlp(mrp['input'])]
                    companions = []
                    length = 0
                    for idx,tokens in enumerate(tokens):
                        token_range = f'TokenRange={length}:{length+len(tokens)}'
                        length += len(tokens)
                        temp = [str(idx+1),tokens,'_','_','_','_','_','_','_',token_range]
                        companions.append(temp)
                    try:
                        mrp["companion"] = companions
                        mrps.append(mrp)
                    except:
                        print(mrp['id'])

                fo.write((json.dumps(mrp) + '\n'))
                line = fi.readline().strip()



if __name__ == '__main__':
    js_path = '/Users/jyq/Desktop/研一/7conll/mrp/2020/split/ucca/ucca_all.mrp'
    out = '/Users/jyq/Desktop/研一/7conll/mrp/2020/split/ucca/ucca_comped.mrp'
    nlp = spacy_udpipe.load('en')
    load_json(8457,js_path,nlp,out)





