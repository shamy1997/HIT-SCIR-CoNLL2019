from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from tqdm import trange
import spacy_udpipe


import collections
import json

nlp = spacy_udpipe.load('en')
# tokenizer = Tokenizer(nlp.vocab)


def load_json(js_path,nlp,output):
    with open(js_path) as fi :
        with open(output,'w') as fo:
            line = fi.readline().strip()
            while line:
                mrp = json.loads(line)
                if 'input' in mrp:
                    tokens, lemmas, pos = [], [], []
                    for t in nlp(mrp['input']):
                        tokens.append(t.text)
                        lemmas.append(t.lemma_)
                        pos.append(t.pos_)
                        companions = []
                        # length = 0
                        init = 0
                    for idx,token in enumerate(tokens):
                        begin = mrp['input'][init:].find(token)
                        # print(mrp['input'][init+begin:init+begin+len(token)])

                        token_range = f'TokenRange={init+begin}:{init+begin+len(token)}'
                        init += begin + len(token)

                        temp = [str(idx+1),token,lemmas[idx],pos[idx],'_','_','_','_','_',token_range]
                        companions.append(temp)
                    try:
                        mrp["companion"] = companions
                    except:
                        print(mrp['id'])



                fo.write((json.dumps(mrp) + '\n'))
                line = fi.readline().strip()



if __name__ == '__main__':
    js_path = '/Users/jyq/Desktop/研一/7conll/mrp/2020/split/eds/eds_all.mrp'
    out = '/Users/jyq/Desktop/研一/7conll/mrp/2020/split/eds/eds_comped.mrp'
    nlp = spacy_udpipe.load('en')
    load_json(js_path,nlp,out)





