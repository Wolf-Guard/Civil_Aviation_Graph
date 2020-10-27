import os
import numpy as np
from collections import Counter, defaultdict

submit_file = 'data_submit/ner_rel/'
ner_rel_group = 'data_submit/three_tuple/'
entity_file = 'data_submit/entity/'
rel_file = 'data_submit/relationship/'


# 结果输出
def read_relation_dict(dict_dir):
    rels_type_list = []
    fname = os.path.join(dict_dir, 'relation_dict.txt')
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            form = []
            category, label = line.strip().split('\t')
            arg1, arg2 = label.split(',')
            arg1 = arg1.split(':')[1]
            arg2 = arg2.split(':')[1]
            form.append(arg1)
            form.append(arg2)
            forms = tuple(form)
            rels_type_list.append(forms)
    return rels_type_list


def read_relation_dict2(dict_dir):
    rels_type_dict = dict()
    fname = os.path.join(dict_dir, 'relation_dict.txt')
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            category, label = line.strip().split('\t')
            arg1, arg2 = label.split(',')
            arg1 = arg1.split(':')[1]
            arg2 = arg2.split(':')[1]
            rels_type_dict.update({(arg1, arg2): category})
    return rels_type_dict


def generate_submission(preds, entity_pairs, dict_dir, threshold):
    doc_rels = defaultdict(set)
    for p, ent_pair in zip(preds, entity_pairs):
        if p > threshold:
            doc_id = ent_pair.doc_id
            f_ent_id = ent_pair.from_ent.ent_id
            t_ent_id = ent_pair.to_ent.ent_id
            category = (ent_pair.from_ent.category, ent_pair.to_ent.category)
            if (category in read_relation_dict(dict_dir)):
                category = read_relation_dict2(dict_dir)[category]
                doc_rels[doc_id].add((f_ent_id, t_ent_id, category))
    submits = dict()
    tot_num_rels = 0
    for doc_id, rels in doc_rels.items():
        output_str = ''
        for i, rel in enumerate(rels):
            tot_num_rels += 1
            line = 'R{}\t{} Arg1:{} Arg2:{}\n'.format(i + 1, rel[2], rel[0], rel[1])
            output_str += line
        submits[doc_id] = output_str
    return submits


def output_submission(submit_file, submits):
    for doc_id, rels_str in submits.items():
        fname = '{}.ann'.format(doc_id)
        filepath = os.path.join(submit_file, fname)
        content = open(filepath, encoding='utf-8').read()
        content += rels_str
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        f.close()


def scan_doc_ids():
    doc_ids = [fname.split('.')[0] for fname in os.listdir(submit_file)]
    doc_ids = [doc_id for doc_id in doc_ids if len(doc_id) > 0]
    return np.unique(doc_ids)


def parse_entity_line(raw_str):
    ents = dict()
    ent_id, label, text = raw_str.strip().split('\t')
    category, pos = label.split(' ', 1)
    pos = pos.split(' ')
    ent = {ent_id: text}
    return ent


def parse_relation_line(raw_str, ents):
    ner_rel = []
    rel_id, label = raw_str.strip().split('\t')
    category, arg1, arg2 = label.split(' ')
    arg1 = arg1.split(':')[1]
    arg2 = arg2.split(':')[1]
    for index in range(len(ents)):
        for ent_id in ents[index].keys():
            if (arg1 == ent_id):
                ner_rel.append(ents[index][ent_id])
                ner_rel.append(category)
            if (arg2 == ent_id):
                ner_rel.append(ents[index][ent_id])
    ner_rel = tuple(ner_rel)
    return ner_rel


def read_anno_file(doc_id):
    ents = []
    ner_rels = []
    fname = os.path.join(submit_file, doc_id + '.ann')
    with open(fname, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('T'):
            ent = parse_entity_line(line)
            ents.append(ent)
    for line in lines:
        if line.startswith('R'):
            ner_rel = parse_relation_line(line, ents)
            ner_rels.append(ner_rel)
    return ner_rels


def get_ent_text(doc_id):
    texts = []
    fname = os.path.join(submit_file, doc_id + '.ann')
    with open(fname, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('T'):
                ent_id, label, text = line.strip().split('\t')
                texts.append(text)
    return texts


# 提取写入实体关系三元组
def get_three_tuple():
    ents = []
    for doc_id in scan_doc_ids():
        fname = '{}.csv'.format(doc_id)
        filepath1 = os.path.join(rel_file, fname)
        filepath2 = os.path.join(entity_file, fname)
        with open(filepath1, 'w', encoding='utf-8') as f:
            f.write('head')
            f.write(',')
            f.write('rel')
            f.write(',')
            f.write('tail')
            f.write('\n')
            for ner_rel in read_anno_file(doc_id):
                for ner in ner_rel:
                    f.write(ner)
                    f.write(',')
                f.write('\n')
        with open(filepath2, 'w', encoding='utf-8') as f:
            f.write('entity')
            f.write(',')
            f.write('\n')
            for entity in get_ent_text(doc_id):
                f.write(entity)
                f.write(',')
                f.write('\n')
