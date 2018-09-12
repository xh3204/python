"""
this model is used to get the influence of the VigilNet
we search the related paper's citation and their authors, according to the
authors's Hindex to reflect the VigilNet's influence
"""
import MySQLdb
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

def get_citation(paperID):
    conn = MySQLdb.connect(host='202.120.36.29', port=3306, user='groupleader', passwd='onlyleaders', db='mag-new-160205', charset="utf8")
    cursor = conn.cursor()
    paperID2citation = 'SELECT PaperID FROM PaperReferences WHERE PaperReferenceID="%s"'
    cursor.execute(paperID2citation % paperID)
    data = cursor.fetchall()
    citation = [i[0] for i in data]
    return citation

def get_citation_count(paperID):
    conn = MySQLdb.connect(host='202.120.36.29', port=3306, user='groupleader', passwd='onlyleaders', db='mag-new-160205', charset="utf8")
    cursor = conn.cursor()
    paperID2citationCount = 'SELECT CitationCount FROM PaperCitationCount WHERE PaperID="%s"'
    cursor.execute(paperID2citationCount % paperID)
    data = cursor.fetchall()
    if data == ():
        citation_count = 0
    else:
        citation_count = data[0][0]
    return citation_count

def get_authors(citation):
    conn = MySQLdb.connect(host='202.120.36.29', port=3306, user='groupleader', passwd='onlyleaders', db='mag-new-160205', charset="utf8")
    cursor = conn.cursor()
    paperID2author = 'SELECT AuthorID FROM PaperAuthorAffiliations WHERE PaperID="%s"'
    authors = []
    for paperID in citation:
        cursor.execute(paperID2author % paperID)
        data = cursor.fetchall()
        authors.extend([i[0] for i in data])
    authors = list(set(authors))
    return authors

def get_Hindex(authors):
    conn = MySQLdb.connect(host='202.120.36.29', port=3306, user='groupleader', passwd='onlyleaders', db='mag-new-160205', charset="utf8")
    cursor = conn.cursor()
    authorID2paper = 'SELECT PaperID FROM PaperAuthorAffiliations WHERE AuthorID="%s"'
    paperID2citation = 'SELECT CitationCount FROM PaperCitationCount WHERE PaperID="%s"'
    Hindex = []
    if not isinstance(authors, list):
        authors = authors.split()
    for authorID in authors:
        citation = []
        cursor.execute(authorID2paper % authorID)
        data = cursor.fetchall()
        paper = [i[0] for i in data]
        for paperID in paper:
            cursor.execute(paperID2citation % paperID)
            data = cursor.fetchall()
            if data != ():
                citation.append(data[0][0])
        citation.sort(reverse = True)
        if citation != []:
            for i in range(len(citation)):
                if citation[i] < i+1:
                    Hindex.append(i)
                    break
                if i == len(citation)-1:
                    Hindex.append(i+1)
        else:
            Hindex.append(0)
        print('complete %d'%(authors.index(authorID)+1))
    return Hindex

print(datetime.now())

citation = get_citation('813AF718')
citation_count = [get_citation_count(i) for i in citation]

# authors = get_authors(citation)
# Hindex = get_Hindex(authors)
# author2Hindex = {}
# for i in range(len(authors)):
#     author2Hindex[authors[i]] = Hindex[i]
# Hindex_dict = {}
# for i in Hindex:
#     if i not in Hindex_dict.keys():
#         Hindex_dict[i] = 1
#     else:
#         Hindex_dict[i] += 1
'''
x = sorted(Hindex_dict.keys())
y = [Hindex_dict[i] for i in x]
name = ['HIndex', 'times']
data = [x, y]
new = np.array(data)
new = list(new.T)
test = pd.DataFrame(columns = name, data = new)
test.to_csv('D:\\学习\\博士\\HIndex.csv')
'''
# leader = [i for i in authors if author2Hindex[i]>=20]
new = np.array(citation_count)
new = list(new.T)
test = pd.DataFrame(columns = ['count'], data = new)
test.to_csv('D:\\学习\\博士\\count.csv')
# print(leader)
print(datetime.now())