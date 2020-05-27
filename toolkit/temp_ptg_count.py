import json
import pandas as pd
from typing import List

from collections import Counter


class PtgInfo():
    def __init__(self,json_file):
        self.sents = []
        self.nodes = []
        self.edges = []
        self.nodes_labels = []
        self.edegs_labels = []
        with open(json_file, 'r', encoding='utf8') as ptg_file:
            for sentence in ptg_file.read().split("\n"):
                try:
                    sent = json.loads(sentence)
                    self.sents.append(sent)
                    self.nodes.append(sent['nodes'])
                    for n in sent['nodes']:
                        if 'label' in n:
                            self.nodes_labels.append(n['label'])
                    self.edges.append(sent['edges'])
                except:
                    pass

    def check_top(self):
        top_edge_labels = {}
        wosmod = 0
        smod = {}


        for idx,edges in enumerate(self.edges):

            for e in edges:

                if e["source"] == 0 and "attributes" not in e:
                    tel = e["label"]

                    if tel in top_edge_labels:
                        top_edge_labels[tel] += 1
                    else:
                        top_edge_labels[tel] = 1

                    next_top_node_idx = e["target"]
                    next_top_node = self.nodes[idx][next_top_node_idx]

                    if "properties" in next_top_node:
                        n_p = next_top_node["properties"]
                        if "sentmod" in n_p:
                            if next_top_node["values"][0] in smod:
                                smod[next_top_node["values"][0]] += 1
                            else:
                                smod[next_top_node["values"][0]] = 1
                    else:
                        wosmod += 1
        return  top_edge_labels
        # 89 句，89个sentmod, 都在次根节点；目前都是enunc ;
        # 根节点可能派生虚节点；

    def check_fullstop_anchoring(self):
        anchor_full_stop = []
        anchor_wo_full_stop = []
        for sent in self.sents:
            for e in sent["edges"]:
                pass




    def check_coref_gram(self):
        two_g, one_g_one_r, two_r = 0,0,0
        co_coref_node = []
        for sent in self.sents:
            for e in sent["edges"]:
                if e["label"] == "coref.gram":
                    begin_idx = e["source"]
                    end_idx = e["target"]
                    begin =  sent["nodes"][begin_idx]["label"]
                    end = sent["nodes"][end_idx]["label"]
                    co_coref_node.append(begin)
                    co_coref_node.append(end)

                    if '#Cor' not in begin and '#Cor' not in end:
                        print(begin,end)

                    if begin.startswith("#") and end.startswith("#"):
                        two_g += 1

                    elif not begin.startswith("#") and not end.startswith("#"):
                        one_g_one_r += 1
                    else:
                        two_r += 1
        print(f'coref.gram: belong to two #node: {two_g} \n'
              f'belong  to one #node and one node: {one_g_one_r} \n'
              f'belong to two node {two_r}')
        co_coref_node = Counter(co_coref_node)
        return co_coref_node


    def check_edge_attr(self):
        attr = {}
        other_edges = []
        all_edges = []
        for sent in self.sents:
            for e in sent["edges"]:
                all_edges.append(e["label"])
                if "attributes" in e:
                    e_label = e["label"]
                    e_attr = e["attributes"][0]
                    if (e_label,e_attr) in attr:
                        attr[(e_label,e_attr)] += 1
                    else:
                        attr[(e_label, e_attr)] = 1
                else:
                    other_edges.append(e["label"])
        other = Counter(other_edges)
        all = Counter(all_edges)

        return (attr,all,other)

        # print(f'label with attr {attr.items()} \n'
        #       f'all edges kinds {all.items()}\n'
        #       f'other edges without edges having attrs {other.items()}')


    def check_anchors_cross_overlap(self):
        count_repeat = 0
        count_cross = 0
        cross_anchors = []
        repeat_anchors = []

        for sent in self.sents:
            anchors = []

            for s in sent['nodes']:
                if "anchors" in s :
                    for an in s["anchors"]:
                        anchor_tuple = (an['from'],an["to"])
                        anchors.append(anchor_tuple)
            anchors = sorted(anchors)
            for i in range(1,len(anchors)):
                if anchors[i][0]< anchors[i-1][1] and anchors[i][0]> anchors[i-1][0]:
                    count_cross += 1
                    cross_anchors.append(sent["id"])
                elif anchors[i][0] == anchors[i-1][0] and anchors[i][1] == anchors[i-1][1]:
                    count_repeat += 1
                    begin = anchors[i][0]
                    end = anchors[i][1]
                    repeat_anchors.append((sent["id"],begin,end,sent["input"][begin:end]))

        print(count_repeat,
              count_cross)
        for id in repeat_anchors:
            print(id)


        # print(f'cross_anchors are {cross_anchors}\n'
        #       f'---------------------------------\n'
        #       f'repeat anchors are {repeat_anchors}')

    # def check_overlap(self):
    #     single_restor_dict = {}
    #     span_restor_dict = {}
    #     overlap_anchors = []
    #
    #     # check whether single nodes have overlap anchoring
    #     for sent in self.sents:
    #         for s in sent['nodes']:
    #             if "anchors" in s and len(s['anchors']) == 1:
    #                 a = s["anchors"][0]
    #                 anchors_str = sent['input'][a['from']:a['to']]
    #                 if anchors_str in





    def check_anchors_type(self):

        type_dict = {}
        multi_anchors = []
        for sent in self.sents:
            for s in sent['nodes']:
                if "anchors" in s and len(s['anchors']) > 1:
                    anchor_str_ls = []
                    for a in s['anchors']:
                        anchors_str = sent['input'][a['from']:a['to']]
                        if anchors_str != s['label']:
                            anchor_str_ls.append(anchors_str)
                            multi_anchors.append(nltk.pos_tag(anchors_str.lower()))

                    type_dict[s["label"]] = anchor_str_ls

                # if 'label' in s and '_' in s['label']:
                #     if len(s['anchors']) == 1:
                #         print(s['label'])

        multi_anchors = Counter(multi_anchors)
        print(multi_anchors.items())


        return type_dict

    def check_slash_nodes_anchors(self):
        slash_nodes = []
        for sent in self.sents:
            for s in sent['nodes']:
                if 'label' in s and s['label'].startswith('#'):
                    slash_nodes.append(s['label'])
        slash_nodes = Counter(slash_nodes)
        return slash_nodes


    def gen_edge_relations(self):
        gedges = []
        for sent in self.sents:
            node_ids = []
            for s in sent["nodes"]:
                if 'label' in s and s['label']=='#Gen':
                    node_ids.append(s["id"])
            node_ids = set(node_ids)
            for e in sent["edges"]:
                if e["source"] in node_ids or e["target"] in node_ids:
                    gedges.append(e['label'])
        gedges = Counter(gedges)
        return gedges


def dict_to_csv(dic,output_csv):
    node_pd = pd.DataFrame.from_dict(dic, orient='index')
    node_pd.to_csv(output_csv)


if __name__ == '__main__':

    file_path = '/Users/jyq/Desktop/研一/7conll/mrp/2020/cf/sample/ptg/wsj.mrp'
    output_1 = '/Users/jyq/Desktop/研一/7conll/mrp/2020/cf/sample/ptg/attr.csv'
    output_2 = '/Users/jyq/Desktop/研一/7conll/mrp/2020/cf/sample/ptg/all.csv'
    output_3 = '/Users/jyq/Desktop/研一/7conll/mrp/2020/cf/sample/ptg/other.csv'
    output_4 = '/Users/jyq/Desktop/研一/7conll/mrp/2020/cf/sample/ptg/anchors.csv'
    output_5 = '/Users/jyq/Desktop/研一/7conll/mrp/2020/cf/sample/ptg/slash_nodes.csv'
    output_6 = '/Users/jyq/Desktop/研一/7conll/mrp/2020/cf/sample/ptg/gen_edges.csv'
    output_7 = '/Users/jyq/Desktop/研一/7conll/mrp/2020/cf/sample/ptg/coref.csv'


    properties = []


    ptg = PtgInfo(file_path)
    # top_labels = ptg.check_top()
    # print(top_labels.items())
    # ptg.check_coref_gram()
    # attr,all,other = ptg.check_edge_attr()
    # dict_to_csv(attr,output_1)
    # dict_to_csv(all,output_2)
    # dict_to_csv(other,output_3)
    # ptg.check_anchors_cross()
    # anchors_type = ptg.check_anchors_type()
    # dict_to_csv(anchors_type,output_4)

    #
    coref_n = ptg.check_coref_gram()
    dict_to_csv(coref_n,output_7)

    # slash_nd = ptg.check_slash_nodes_anchors()
    # dict_to_csv(slash_nd,output_5)
    # print(len(ptg.nodes_labels))

    # print(ptg.check_anchors_cross_overlap())
    # gen_edges = ptg.gen_edge_relations()
    # dict_to_csv(gen_edges,output_6)
