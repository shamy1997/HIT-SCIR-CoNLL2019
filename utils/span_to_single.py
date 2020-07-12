from tqdm import tqdm
import json

SYMBOLS = {'(':'#bracket','&':"#amp","%": "#Percnt", "*":"#Ast", ":": "#Colon", "…": "#Period3", ",":"#Comma", "—":"#Dash","/":"#Slash",";":'#Semicolon'}


def load_mrp(mrp):
    raw_mrp = json.loads(mrp)
    nodes = raw_mrp["nodes"]
    if len(nodes) < 3:
        return False
    dependent = raw_mrp["companion"][1]
    dependent_dict = {}
    for depend in dependent:
        dependent_dict[depend["target"]] = depend

    companion_list = raw_mrp["companion"][0]
    companion = {}
    for idx,comp in enumerate(companion_list):
        try:
            for key,value in comp.items():
                companion[key] = value
            if idx in dependent_dict:
                companion[key]["dep"] = dependent_dict[idx]["label"]
            else:
                companion[key]["dep"] = ''

        except AttributeError:
            continue

    single_potentials = []
    gold_spans = []
    for idx,node in enumerate(nodes):
        single_tokens = []

        if idx == 0:
            continue
        try:
            if len(node["anchors"]) > 1:
                anchors = [f'{anchor["from"]}:{anchor["to"]}' for anchor in node["anchors"]]
                gold_span = node["label"]
                for anchor in anchors:
                    if companion[anchor]["lemma"] != '.' and companion[anchor]["lemma"] != '"':
                        single_tokens.append(companion[anchor])

                single_potentials.append(single_tokens)

                gold_spans.append(gold_span)
        except KeyError:
            continue
    return (gold_spans,single_potentials)


def single(single_tokens):
    if single_tokens[0]["lemma"] in SYMBOLS:
        return [SYMBOLS[single_tokens[0]["lemma"]]]
    else:
        return [single_tokens[0]["lemma"]]

def symbol_filter(single_tokens):
    pred = []
    for token in single_tokens:
        if token["upos"] == "SYM" :
            if token["lemma"] in SYMBOLS:
                pred.append(SYMBOLS[token["lemma"]])
            else:
                pred.append(token["lemma"])
    return pred



def pos_filter(single_tokens):
    pred = []
    for token in single_tokens:
        if token["upos"] == "NOUN"  or \
                token["xpos"] == "NNP" or \
                token["xpos"] == "RB" and token["upos"] != "ADV" or \
                token["xpos"] == "RB" and token["dep"] == "cc" or \
                token["xpos"] == "RB" and token["dep"] == "advmod" or\
                token["upos"] == "VERB" and token["dep"] != "case" or \
                token["upos"] == "ADJ" or \
                token["upos"] == "NUM" or \
                token["upos"] == "X" or \
                "comp" in token["dep"] or\
                token["dep"] == "obl":
            pred.append(token["lemma"])

    return pred


def match_twice(single_tokens):
    pred = []
    for token in single_tokens:
        if token["upos"] == "PRON" and token["xpos"] == "PRP":
            return ['#PersPron']
        elif token["upos"] == "PROPN":
            return [token["lemma"]]
        elif token["xpos"] == "RB":
            return [token["lemma"]]
        elif token["upos"] == "CCONJ" or\
                token["dep"] == "fixed":
            pred.append(token["lemma"])
    if pred:
        return pred
    else:
        return [single_tokens[0]["lemma"]]


def dep_filter(single_tokens,pred):
    for token in single_tokens:
        if "dep" in token:
            if token["dep"] == "fixed":
                pred.append(token["lemma"])
    return pred


def run(mrp):
    if load_mrp(mrp):
        gold_spans,single_potentials = load_mrp(mrp)
    else:
        return
    preds_spans = []
    match  = 0
    for single_tokens in single_potentials:
        if len(single_tokens) == 1:
            pred = single(single_tokens)
        else:
            pred = symbol_filter(single_tokens)
        if not pred:
            pred = pos_filter(single_tokens)
        # pred = dep_filter(single_tokens,pred)
        if not pred:
            pred = match_twice(single_tokens)
        pred = '_'.join(pred)
        preds_spans.append(pred)
    for idx,pred in enumerate(preds_spans):
        if gold_spans[idx].lower() == pred.lower():
            match += 1
        # elif gold_spans[idx] == "#Dash":
        #     print('ok')
        #     exit(0)
        else:
            print(gold_spans[idx].lower(),pred.lower())

    return (len(gold_spans),match)


if __name__ == '__main__':
    mrp = '/Users/jyq/Desktop/研一/7conll/mrp/2020/split/ptg/span_anchors/ptg_train_4000.aug.mrp'
    with open(mrp) as fi:
        mrp_lines = [line.strip() for line in fi.readlines()]
        all = 0
        match = 0
        a = 0
        for mrp_line in tqdm(mrp_lines):
            a+=1
            # # print(a)
            if run(mrp_line):
                gold_num , match_num = run(mrp_line)
            else:
                continue
            all += gold_num
            match += match_num


        #debug
        # gold_num,match_num = run(mrp_lines[3981])

    print(match/all)

