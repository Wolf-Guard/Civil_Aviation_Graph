from py2neo import Graph
from py2neo import Node
from py2neo import Relationship
import pandas as pd
import time

def gen_subg(rels):
    subg = rels[0]
    for rel in rels:
        subg = subg | rel
    return subg

def create_nodes(nodes):
    nodes_dict = {}
    for node in nodes:
        nodes_dict[node] = Node(node, name=node)
    return nodes_dict


def create_rels(rels):
    rel_dict = {}
    for rel in rels:
        rel_dict[rel] = Relationship.type(rel)
    return rel_dict


def make_rel(node_dict, rel_dict, row):
    head, rel, tail = row[0], row[1], row[2]
    return rel_dict[rel](node_dict[head], node_dict[tail])

def create_graph():
    print("数据读取中.....")
    df = pd.read_csv("data_submit/relationship/0.csv", encoding="utf-8", names=["head", "rel", "tail"])

    print("读取节点关系数据中.....")
    heads, tail = df['head'], df['tail']
    rels = df['rel']
    nodes = set(heads.tolist() + tail.tolist())
    rels = set(df['rel'])
    gnodes_dict = create_nodes(nodes)
    grels_dict = create_rels(rels)

    i = 0
    relations = []
    print("连接neo4j数据库中....")
    graph = Graph('http://localhost:7474', username='neo4j', password='xmh885202')

    print("导入图谱节点中....")
    start = time.time()
    print('导入图谱边中....')
    for row in df.iterrows():
        lstart = time.time()
        i = i + 1
        relation = make_rel(gnodes_dict, grels_dict, row[1])
        relations.append(relation)
        if i % 100 == 0:
            subg = gen_subg(relations)
            graph.create(subg)
            relations = []
            print(i, "rows are written, time spent, total:", time.time() - start, "loop:", time.time() - lstart)

