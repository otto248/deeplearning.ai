#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:10:53 2018

@author: cuiym
"""

#读数据库
#返回三元组格式
import pymysql
import pandas as pd
import re

def read_database():
    db = 'momokb2'
    config = {
              'host':'10.3.13.4',
              'port':3306,
              'user':'root',
              'password':'1qazZAQ!',
              'db':db,
              'charset':'utf8',
              } 
    con = pymysql.connect(**config)
    sql1 = "select * from KG_yy"
    raw_infor_knowledge_graph = pd.read_sql(sql1, con)
    sql2 = "select * from KG_relation_weight"
    raw_infor_weight = pd.read_sql(sql2, con)
    
    collection = []
    for index in raw_infor_knowledge_graph.index:
        entity_main_node = raw_infor_knowledge_graph.loc[index]['knowledge_node']
        entity_sub_node = raw_infor_knowledge_graph.loc[index]['sub_knowledge_node']
        if entity_main_node not in collection:
            collection.append(entity_main_node)
        if entity_sub_node not in collection:
            collection.append(entity_sub_node)
    
    return [raw_infor_knowledge_graph,raw_infor_weight,collection]

#输入字符串的模糊处理
#输入匹配的字符串，和要匹配到的所有字符组成的列表
#返回匹配到的字符串列表
def fuzzymatch(usr_input,collection):
    suggestions = []
    pattern = '.*?'.join(usr_input)
    regex = re.compile(pattern)
    for item in collection:
        match = regex.search(item)
        if match:
            suggestions.append((len(match.group()), match.start(), item))
    return [x for _, _, x in sorted(suggestions)]

#输入原始数据三元组
#输入匹配到的实体列表
#输入数据库的权重表
def search(knowledge_graph, matchEntity,df_relation_weight,relationString):
    #初始化结构
    recmd = pd.DataFrame()
    matchEntity = matchEntity[0]
    #先利用职级缩小人的遍历
    matchEntity_sub_node = knowledge_graph[knowledge_graph.knowledge_node == matchEntity]
    leader = matchEntity_sub_node[matchEntity_sub_node.relation == '职级'].iloc[0]['sub_knowledge_node']
    knowledge_graph_person = knowledge_graph.loc[(knowledge_graph.knowledge_node==leader)&(knowledge_graph.relation=='组员')]
    if not knowledge_graph.loc[(knowledge_graph.knowledge_node==leader)&(knowledge_graph.relation=='上级')].empty:
        knowledge_graph_person_2 = knowledge_graph.loc[(knowledge_graph.knowledge_node==leader)&(knowledge_graph.relation=='上级')]
        knowledge_graph_person = pd.concat([knowledge_graph_person,knowledge_graph_person_2])
    else:
        pass
    
    if relationString == "1":
        recmd = knowledge_graph_person
    else:
        #遍历做近似计算的特征
        relationList = relationString.split(',')#['荣耀','年龄'] 
        for relation in relationList:
            recmd_relation = pd.DataFrame()
            matchEntity_relation = matchEntity_sub_node[matchEntity_sub_node.relation == relation]
            weight = df_relation_weight[df_relation_weight.relation == relation].iloc[0]['weight']        
            i = 0
            for person in knowledge_graph_person[knowledge_graph_person.sub_knowledge_node != matchEntity].sub_knowledge_node.value_counts().index:
                df_person = knowledge_graph[knowledge_graph.knowledge_node == person]
                df_character = df_person[df_person.relation == relation]
                multi_weight = len(pd.merge(df_character,matchEntity_relation,on='sub_knowledge_node')) * weight
                df_recmd_person = pd.DataFrame({'person':person,relation:multi_weight},index = [i])  
                recmd_relation = pd.concat([recmd_relation, df_recmd_person])
                i += 1
            if recmd.empty:
                recmd = pd.concat([recmd,recmd_relation])
            else:
                recmd = pd.merge(recmd,recmd_relation,on='person')

        #对各个特征加权  
        recmd['Sum_weight'] = recmd.iloc[:,1:].apply(lambda x:x.sum(),axis=1)
        recmd = recmd.nlargest(20,columns='Sum_weight')
    return recmd


if __name__ == '__main__':
    raw_info = read_database()
    matchEntity = fuzzymatch('',raw_info[2])
    df_search = search(raw_info[0],matchEntity,raw_info[1])
    final_result = df_search.to_json(orient='records')
   
