''' 
    this file can return one's  highest ranked field, and it may be more than one. if you
    want to use it, please use the function 'author2venue'. you just need to give it a n-
    ame id as the input, and then it will return the filed list.
'''

import MySQLdb
from datetime import datetime

def author2filed(authorID):
    conn = MySQLdb.connect(host='202.120.36.29', port=3306, user='groupleader', passwd='onlyleaders', db='mag-new-160205',
                            charset="utf8")
    cursor = conn.cursor()
    author2field_rank = 'SELECT FieldId, Rank FROM AuthorFieldRank WHERE AuthorID="%s"'
    fieldID2field_name = 'SELECT FieldsOfStudyName FROM FieldsOfStudy WHERE FieldsOfStudyID="%s"'
    cursor.execute(author2field_rank % authorID)
    field_rank = cursor.fetchall()
    field2rank = {}
    for i in field_rank:
        fieldID = i[0]
        rank = i[1]
        cursor.execute(fieldID2field_name % fieldID)
        field_name = cursor.fetchall()[0][0]
        field2rank[field_name] = rank
    rank2field = {}
    for key, value in field2rank.items():
        if value not in rank2field.keys():
            rank2field[value] = []
            rank2field[value].append(key)
        else:
            rank2field[value].append(key)
    # sort the rank from big to small
    rank_sort = sorted(list(rank2field.keys()))

    return field2rank

print(author2filed('7EA9B3E8'))