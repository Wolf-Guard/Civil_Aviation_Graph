from train import *
from data_extractor import *
from Create_Graph import *
from QA import *
from keras.models import load_model

#模型调用
my_model = load_model('Model/model.h5')
print('模型调用成功！')

#模型预测
te_sent, te_ent, te_f_ent, te_t_ent, te_ent_dist, te_y = test_data[:]
preds = my_model.predict(x=[te_sent, te_ent, te_f_ent, te_t_ent, te_ent_dist], verbose=1)

#结果输出
submit_file = 'data_submit/ner_rel/'
submits = generate_submission(preds, test_entity_pairs_clean,relation_dict_dir,0.5)
output_submission(submit_file, submits)

#获取实体关系csv
get_three_tuple()

#创建知识图谱
create_graph()

#对话机器人
QA()