# coding:utf-8
import os
from lxml import html
import re
import MySQLdb
import spacy
import numpy as np
import datetime
import math

stopwords = set()
for line in open('./stopwords_ace.txt'):
    stopwords.add(line.replace('\n', '').replace('\r', ''))
# print stopwords

# 用到了spacy预先训练好的词向量（https://spacy.io）
# nlp = spacy.load('en_core_web_md')
# print 'spacy.load finished'


conn = MySQLdb.connect(host='202.120.36.29', port=6033, user='groupleader', passwd='onlyleaders', db='mag-new-160205',
                       charset="utf8")
cursor = conn.cursor()


class Paper:
    def __init__(self, paper_id, title, label):
        self.paper_id = paper_id
        self.title = title
        # self.year = year
        # self.venue_id = venue_id
        # self.affiliation_id = affiliation_id
        # self.coauthors = coauthors
        self.label = label
        self.label_predicted = 0

        title_nlp = nlp(unicode(title.encode('utf-8').decode('utf-8')))

        title_vector_sum = np.zeros(title_nlp[0].vector.shape)
        word_count = 0
        self.title_vector = np.zeros(title_nlp[0].vector.shape)

        for word in title_nlp:
            if str(word) not in stopwords and len(str(word)) > 1:
                title_vector_sum += word.vector
                word_count += 1
        if word_count != 0:
            self.title_vector = title_vector_sum / word_count

# 一个文章簇，包含了文章列表，文章index列表，机构列表，发表年份，在同一年份发表的所有机构
class Cluster:
    def __init__(self, paper, paper_idx, affiliation_id, year):
        # paper = papers[idx]
        self.papers = list()
        self.papers.append(paper)
        self.paper_idx_list = list()
        self.paper_idx_list.append(paper_idx)
        self.cluster_id = paper_idx
        self.affiliations = set()
        self.affiliations.add(affiliation_id)
        self.year_2_affiliations = dict()
        if affiliation_id is not None and year is not None:
            self.year_2_affiliations[year] = set()
            self.year_2_affiliations[year].add(affiliation_id)

        self.link_type_2_ngbrs = dict()
        self.ngbrs = set()
    # 将两个簇合为一个
    def unit(self, other, paper_idx_2_cluster_id):
        # 把另一个簇中所有paper_idx所对应的cluster_id都改为这个簇的cluster_id
        for paper_idx in other.paper_idx_list:
            paper_idx_2_cluster_id[paper_idx] = self.cluster_id
        self.papers.extend(other.papers)
        self.paper_idx_list.extend(other.paper_idx_list)
        self.affiliations |= other.affiliations
        # 将两个簇年份2机构的字典合并
        for k, v in other.year_2_affiliations.iteritems():
            if k in self.year_2_affiliations.keys():
                self.year_2_affiliations[k] |= v
            else:
                self.year_2_affiliations[k] = v

    # 判断两个簇是否能合并,False则冲突
    def has_no_conflict(self, other, paper_final_edges, strict_mode):
        connected_edges = 0
        for paper_idx in other.paper_idx_list:
            connected_edges += len(np.nonzero(paper_final_edges[paper_idx, self.paper_idx_list])[0])

        if strict_mode and float(connected_edges) < 0.01 * (len(self.papers) * len(other.papers)):
            return False

        if len(self.affiliations | other.affiliations) > 20:
            return False

        for k, v in self.year_2_affiliations.iteritems():
            if k in other.year_2_affiliations.keys():
                if len(v | other.year_2_affiliations[k]) > 3:
                    return False

        return True

# 将一个或多个文件路径放到file_list中
def get_file_list(dir, file_list):
    newDir = dir
    if os.path.isfile(dir):
        file_list.append(dir.decode('gbk'))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            get_file_list(newDir, file_list)
    return file_list


def compute_title_similarity(paper_A, paper_B):
    """calculate title similarity between tow papers

    Returns
    --------
    cos_sim : corresponding similarity of tow papers
    """
    vector_A = paper_A.title_vector
    vector_B = paper_B.title_vector

    # 可改进，直接将vector和np.zeros(300)比较
    if len(np.nonzero(vector_A)[0]) == 0 or len(np.nonzero(vector_B)[0]) == 0:
        return 0
    cos_sim = np.dot(vector_A, vector_B) / (np.linalg.norm(vector_A) * np.linalg.norm(vector_B))
    return cos_sim


def link_type_2_weight(link_type, except_type):
    """get corresponding weight according to 'link_type'

    the weight is used to calculate shortest path, so the smaller weight means more confident
    if the 'link_type' is the same as 'except_type', returns higher weight

    Returns
    --------
    corresponding weight of 'link_type'

    """
    if link_type == 's':  # strong connection
        return 0

    if link_type[-1] == 'w' and link_type[:-1] == except_type:
        # the same type as 'except_type' but with strong content relevancy
        if link_type[0] == 'a':  # co-author
            return 4
        elif link_type[0] == 'o':  # co-affiliation
            return 5
        elif link_type[0] == 'v':  # co-venue
            return 6
    else:
        if link_type[0] == 'a':  # co-author
            return 1
        elif link_type[0] == 'o':  # co-affiliation
            return 2
        elif link_type[0] == 'v':  # co-venue
            return 3


def add_in_inverted_indices(inverted_indices, paper_idx, feature_uni_id):
    if feature_uni_id not in inverted_indices:
        inverted_indices[feature_uni_id] = list()
    inverted_indices[feature_uni_id].append(paper_idx)


def analyze_papers_and_init_clusters(file):
    author_name = file.split('/')[-1].replace('.xml', '')
    author_name = author_name.lower()
    author_name = re.sub('[^A-Za-z0-9]', ' ', author_name)
    author_name = re.sub('\s{2,}', ' ', author_name)

    print author_name,

    tree = html.parse(file)
    root = tree.getroot()  # 获取根节点

    process_count = 0
    papers = list()
    clusters = dict()
    paper_idx_2_cluster_id = dict()
    inverted_indices = dict()

    uni_id_generator = 0
    coauthor_2_uni_id = dict()
    affiliation_2_uni_id = dict()
    venue_2_uni_id = dict()

    # # remove authors who have only one paper
    # label_2_papers = dict()
    # for node in root.xpath('//publication'):
    #     label = node.xpath('label')[0].text
    #     paper_id = node.xpath('id')[0].text

    #     if label not in label_2_papers:
    #         label_2_papers[label] = set()
    #     label_2_papers[label].add(paper_id)

    # papers_reserved = set()
    # for label, label_papers in label_2_papers.iteritems():
    #     if len(label_papers) > 1:
    #         for paper_id in label_papers:
    #             papers_reserved.add(paper_id)


    for node in root.xpath('//publication'):
        original_paper_id = node.xpath('id')[0].text
        # if original_paper_id not in papers_reserved:
        #     continue

        label = node.xpath('label')[0].text
        title = node.xpath('title')[0].text

        title = title.lower()
        if title[-1] == '.':
            title = title[:-1]
        title = re.sub('[^A-Za-z0-9]', ' ', title)
        title = re.sub('\s{2,}', ' ', title)
        quest_paper_by_title = 'SELECT PaperID FROM Papers WHERE NormalizedPaperTitle="%s"'
        cursor.execute(quest_paper_by_title % title)
        ps = cursor.fetchall()

        paper_ids = list()
        if len(ps) == 1:
            paper_ids.append(ps[0][0])
        if len(ps) >= 2:
            for p in ps:
                quest_author_by_paper = 'SELECT AuthorName FROM Authors INNER JOIN' \
                                        '   (SELECT AuthorID FROM PaperAuthorAffiliations AS PAA  WHERE PaperID="%s") AS TB2' \
                                        '   ON Authors.AuthorID = TB2.AuthorID'
                cursor.execute(quest_author_by_paper % p[0])
                authors = cursor.fetchall()
                for author in authors:
                    if author[0] == author_name.lower():
                        paper_ids.append(p[0])

        for paper_id in paper_ids:
            paper_idx = process_count

            # get affiliation and coauthors
            quest_affiliation = 'SELECT AuthorName,AffiliationID FROM Authors INNER JOIN' \
                                '   (SELECT AuthorID,AffiliationID FROM PaperAuthorAffiliations WHERE PaperID="%s") AS TB ' \
                                'ON Authors.AuthorID = TB.AuthorID'
            cursor.execute(quest_affiliation % paper_id)
            author_affiliations = cursor.fetchall()

            himself = None
            for ai in range(len(author_affiliations)):
                if author_affiliations[ai][0] == author_name.lower():
                    himself = ai
                    break

            if himself is None:
                tmp1 = author_name.split()
                count = 0
                for ai in range(len(author_affiliations)):
                    tmp2 = author_affiliations[ai][0].split()
                    if tmp1[-1] == tmp2[-1] and tmp1[0][0] == tmp2[0][0]:
                        himself = ai
                        break
                    elif tmp1[-1] == tmp2[0] and tmp1[0][0] == tmp2[-1][0]:
                        himself = ai
                        break

            # get affiliation
            if himself is None:
                #                 affiliation_id = author_affiliations[-1][1]
                affiliation_id = None
            else:
                affiliation_id = author_affiliations[himself][1]
            if affiliation_id is not None:
                if affiliation_id not in affiliation_2_uni_id:
                    affiliation_2_uni_id[affiliation_id] = 'o' + str(uni_id_generator)
                    uni_id_generator += 1
                affiliation_id = affiliation_2_uni_id[affiliation_id]

                add_in_inverted_indices(inverted_indices, paper_idx, affiliation_id)

            # get coauthors
            coauthors = set()
            for ai in range(len(author_affiliations)):
                if ai != himself:
                    coauthor_name = author_affiliations[ai][0]
                    if coauthor_name not in coauthor_2_uni_id:
                        coauthor_2_uni_id[coauthor_name] = 'a' + str(uni_id_generator)
                        uni_id_generator += 1
                    # coauthors.add(coauthor_2_uni_id[coauthor_name])
                    add_in_inverted_indices(inverted_indices, paper_idx, coauthor_2_uni_id[coauthor_name])

            # get venue, title and year
            venue_id = None
            year = None
            quest_info_by_paper = 'SELECT NormalizedPaperTitle, ConferenceSeriesIDMappedToVenueName, ' \
                                  'JournalIDMappedToVenueName, PaperPublishYear FROM Papers WHERE PaperID = "%s"'
            cursor.execute(quest_info_by_paper % paper_id)
            rs = cursor.fetchall()
            if len(rs) != 0:
                # fill in paper_venue_dict
                if rs[0][1] is not None:
                    venue_id = rs[0][1]
                elif rs[0][2] is not None:
                    venue_id = rs[0][2]

                if venue_id is not None:
                    if venue_id not in venue_2_uni_id:
                        venue_2_uni_id[venue_id] = 'v' + str(uni_id_generator)
                        uni_id_generator += 1
                    venue_id = venue_2_uni_id[venue_id]
                    add_in_inverted_indices(inverted_indices, paper_idx, venue_id)

                year = rs[0][3]

            paper_instance = Paper(paper_id, title, label)
            papers.append(paper_instance)

            # initially each paper is used as a cluster
            new_cluster = Cluster(paper_instance, paper_idx, affiliation_id, year)
            clusters[paper_idx] = new_cluster
            paper_idx_2_cluster_id[paper_idx] = paper_idx
            process_count += 1

    print '\t' + str(len(papers)),
    return papers, clusters, paper_idx_2_cluster_id, inverted_indices


def init_paper_edges_and_ngbrs(papers, inverted_indices):
    paper_count = len(papers)

    title_sim_matrix = np.zeros((paper_count, paper_count))
    paper_full_edges = [[0 for col in range(paper_count)] for row in range(paper_count)]
    paper_all_ngbrs = [set() for i in range(paper_count)]
    paper_weak_type_ngbrs = [dict() for i in range(paper_count)]
    cluster_merge_pairs = list()

    paper_weak_edges = set()
    paper_strong_edges = set()
    paper_tmp_edges = [[0 for col in range(paper_count)] for row in range(paper_count)]

    for i in range(paper_count):
        for j in range(i + 1, paper_count):
            title_sim = compute_title_similarity(papers[i], papers[j])
            title_sim_matrix[i, j] = title_sim
            title_sim_matrix[j, i] = title_sim

    for link_type, paper_list in inverted_indices.iteritems():
        for i in range(len(paper_list)):
            paper_i = paper_list[i]
            for j in range(i + 1, len(paper_list)):
                paper_j = paper_list[j]

                if paper_tmp_edges[paper_i][paper_j] == 0:
                    paper_tmp_edges[paper_i][paper_j] = link_type
                    paper_tmp_edges[paper_j][paper_i] = link_type
                    paper_weak_edges.add((paper_i, paper_j))
                    paper_full_edges[paper_i][paper_j] = link_type
                    paper_full_edges[paper_j][paper_i] = link_type
                    paper_all_ngbrs[paper_i].add(paper_j)
                    paper_all_ngbrs[paper_j].add(paper_i)

                elif paper_tmp_edges[paper_i][paper_j] != -1:  # strong connection
                    cluster_merge_pairs.append(set((paper_i, paper_j)))
                    paper_strong_edges.add((paper_i, paper_j))
                    paper_tmp_edges[paper_i][paper_j] = -1
                    paper_tmp_edges[paper_j][paper_i] = -1
                    paper_full_edges[paper_i][paper_j] = 's'
                    paper_full_edges[paper_j][paper_i] = 's'

    paper_weak_edges = paper_weak_edges - paper_strong_edges
    for weak_edge in paper_weak_edges:
        i = weak_edge[0]
        j = weak_edge[1]
        link_type = paper_full_edges[i][j]

        if link_type not in paper_weak_type_ngbrs[i].keys():
            paper_weak_type_ngbrs[i][link_type] = set()
        paper_weak_type_ngbrs[i][link_type].add(j)
        if title_sim_matrix[i, j] > 0.5:
            paper_full_edges[i][j] = link_type + 'w'
            paper_full_edges[j][i] = link_type + 'w'

    return paper_full_edges, paper_all_ngbrs, paper_weak_type_ngbrs, \
           cluster_merge_pairs, title_sim_matrix


def merge_strong_connected_papers(clusters, paper_idx_2_cluster_id, cluster_merge_pairs):
    has_changed = True
    while has_changed:
        has_changed = False
        pair_num = len(cluster_merge_pairs)
        i = 0
        while i < pair_num:
            j = i + 1
            while j < pair_num:
                if len(cluster_merge_pairs[i] & cluster_merge_pairs[j]):
                    cluster_merge_pairs[i] = cluster_merge_pairs[i] | cluster_merge_pairs[j]
                    cluster_merge_pairs.remove(cluster_merge_pairs[j])
                    pair_num -= 1
                    j = i + 1
                    has_changed = True
                else:
                    j += 1
            i += 1

    for merge in cluster_merge_pairs:
        A = merge.pop()
        for B in merge:
            clusters[A].unit(clusters[B], paper_idx_2_cluster_id)
            del clusters[B]
    return clusters


def generate_cluster_edges(clusters, papers,paper_full_edges, paper_weak_type_ngbrs, paper_idx_2_cluster_id):
    # sort cluster by number of papers
    sorted_clusters = sorted(clusters.iteritems(), key=lambda d: len(d[1].papers), reverse=True)

    # change clusters' type(dict) to list
    clusters = list()
    for c in range(len(sorted_clusters)):
        cluster = sorted_clusters[c][1]
        cluster.cluster_id = c
        for paper_idx in cluster.paper_idx_list:
            paper_idx_2_cluster_id[paper_idx] = c
        clusters.append(cluster)

    # initialize cluster edges
    cluster_initial_edges = [[set() for col in range(len(clusters))] for row in range(len(clusters))]

    for i in range(len(papers)):
        cluster_i = paper_idx_2_cluster_id[i]
        for i_link_type, i_ngbrs in paper_weak_type_ngbrs[i].iteritems():
            papers_in_same_cluster = set(clusters[paper_idx_2_cluster_id[i]].paper_idx_list)  # including itself
            i_ngbrs -= papers_in_same_cluster
            if len(i_ngbrs) == 0:
                continue

            for j in i_ngbrs:
                cluster_j = paper_idx_2_cluster_id[j]
                cluster_initial_edges[cluster_i][cluster_j].add(paper_full_edges[i][j])
                cluster_initial_edges[cluster_j][cluster_i].add(paper_full_edges[i][j])
                clusters[cluster_i].ngbrs.add(cluster_j)
                clusters[cluster_j].ngbrs.add(cluster_i)
    # print cluster_edges

    # generate clusters' link_type_2_ngbrs
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if len(cluster_initial_edges[i][j]) != 0:
                for link_type in cluster_initial_edges[i][j]:
                    if link_type[-1] == 'w':
                        link_type = link_type[:-1]
                    if link_type not in clusters[i].link_type_2_ngbrs:
                        clusters[i].link_type_2_ngbrs[link_type] = set()
                    clusters[i].link_type_2_ngbrs[link_type].add(j)

    cluster_final_edges = [[dict() for col in range(len(clusters))] for row in range(len(clusters))]

    INFINITY = 9999
    for i in range(len(clusters)):
        for i_link_type, i_ngbrs in clusters[i].link_type_2_ngbrs.iteritems():

            # Dijkstra's algorithm
            S = set()  # visited
            Q = set(range(len(clusters)))  # unvisited

            dist = np.array([INFINITY] * len(papers))
            dist_final = np.array([INFINITY] * len(papers))

            S.add(i)
            dist[i] = 0

            while len(Q) != 0:

                u = dist.argmin()

                if dist[u] > 18:
                    break

                Q.remove(u)
                S.add(u)
                dist_final[u] = dist[u]

                if len(i_ngbrs - S) == 0:
                    break

                for v in clusters[u].ngbrs:
                    if v in Q:
                        uv_link_types = cluster_initial_edges[u][v]
                        for uv_link_type in uv_link_types:
                            if uv_link_type == i_link_type or (uv_link_type[0] == 'v' and i_link_type[0] == 'v'):
                                continue
                            alt = dist[u] + link_type_2_weight(uv_link_type, i_link_type)
                            if alt < dist[v]:
                                dist[v] = alt
                dist[u] = INFINITY

            for i_ngbr in i_ngbrs:
                if dist_final[i_ngbr] != INFINITY and dist_final[i_ngbr] != 0:
                    weight = 1.0 / dist_final[i_ngbr]
                    cluster_final_edges[i][i_ngbr][i_link_type] = weight
                    cluster_final_edges[i_ngbr][i][i_link_type] = weight
                    # cluster_similarity_dict[tuple((i, i_ngbr, i_link_type))] = weight

    # change 'cluster'(list) to dict
    tmp_clusters = dict()
    for i in range(len(clusters)):
        tmp_clusters[i] = clusters[i]
    clusters = tmp_clusters

    return clusters, cluster_final_edges


def generate_paper_similarity_dict(papers, paper_idx_2_cluster_id, paper_weak_type_ngbrs, cluster_edges):
    paper_similarity_dict = dict()

    paper_final_edges = np.zeros((len(papers), len(papers)))

    for i in range(len(papers)):

        cluster_i_id = paper_idx_2_cluster_id[i]

        for i_link_type, i_ngbrs in paper_weak_type_ngbrs[i].iteritems():
            for j in i_ngbrs:
                cluster_j_id = paper_idx_2_cluster_id[j]
                if cluster_i_id == cluster_j_id:
                    continue
                if i_link_type in cluster_edges[cluster_i_id][cluster_j_id]:
                    weight = cluster_edges[cluster_i_id][cluster_j_id][i_link_type]
                    paper_final_edges[i, j] = weight
                    paper_final_edges[j, i] = weight
                    paper_similarity_dict[tuple((i, j))] = weight
    return paper_similarity_dict, paper_final_edges


def hierarchical_clustering(paper_similarity_dict, paper_final_edges, clusters, paper_idx_2_cluster_id):
    sorted_similarity_pairs = sorted(paper_similarity_dict.iteritems(), key=lambda d: d[1], reverse=True)

    for pair in sorted_similarity_pairs:
        paper_A_idx = pair[0][0]
        paper_B_idx = pair[0][1]
        similarity = pair[1]

        cluster_A_id = paper_idx_2_cluster_id[paper_A_idx]
        cluster_B_id = paper_idx_2_cluster_id[paper_B_idx]
        if cluster_A_id == cluster_B_id:
            continue

        cluster_A = clusters[cluster_A_id]
        cluster_B = clusters[cluster_B_id]

        if cluster_A != cluster_B:
            cluster_small = cluster_A
            cluster_big = cluster_B
            cluster_small_id = cluster_A_id
            if len(cluster_A.papers) > len(cluster_B.papers):
                cluster_small = cluster_B
                cluster_big = cluster_A
                cluster_small_id = cluster_B_id

            if cluster_big.has_no_conflict(cluster_small, paper_final_edges, True):
                cluster_big.unit(cluster_small, paper_idx_2_cluster_id)
                del clusters[cluster_small_id]
    return clusters


def merge_scattered_papers(clusters, paper_idx_2_cluster_id, title_sim_matrix, paper_all_ngbrs, paper_final_edges):
    cluster_merge_pairs = list()
    for cluster_id, cluster in clusters.iteritems():
        if len(cluster.papers) == 1:
            paper_idx = cluster.paper_idx_list[0]
            top_indices = np.argsort(-title_sim_matrix[paper_idx, :])
            for i in top_indices:
                if title_sim_matrix[paper_idx, i] > 0.8 or (
                                title_sim_matrix[paper_idx, i] > 0.4 and i in paper_all_ngbrs[paper_idx]):
                    cluster_small_id = paper_idx_2_cluster_id[paper_idx]
                    cluster_big_id = paper_idx_2_cluster_id[i]
                    if len(clusters[cluster_big_id].papers) > 5 \
                            and clusters[cluster_big_id].has_no_conflict(clusters[cluster_small_id],
                                                                         paper_final_edges,
                                                                         False):
                        cluster_merge_pairs.append((cluster_small_id, cluster_big_id))
                        break

    for merge in cluster_merge_pairs:
        clusters[merge[1]].unit(clusters[merge[0]], paper_idx_2_cluster_id)
        del clusters[merge[0]]

    return clusters


def name_disambiguation(file):
    # analyze papers and initialize clusters
    papers, clusters, paper_idx_2_cluster_id, inverted_indices = analyze_papers_and_init_clusters(file)

    # initialize papers' edges and ngbrs
    paper_full_edges, paper_all_ngbrs, paper_weak_type_ngbrs, \
    cluster_merge_pairs, title_sim_matrix = init_paper_edges_and_ngbrs(papers, inverted_indices)

    # merge strong connected papers
    clusters = merge_strong_connected_papers(clusters, paper_idx_2_cluster_id, cluster_merge_pairs)

    # generate cluster edges
    clusters, cluster_edges = generate_cluster_edges(clusters, papers, paper_full_edges, paper_weak_type_ngbrs, paper_idx_2_cluster_id)

    # generate paper similarity dict
    paper_similarity_dict, paper_final_edges \
        = generate_paper_similarity_dict(papers, paper_idx_2_cluster_id, paper_weak_type_ngbrs, cluster_edges)

    # hierarchical clustering
    clusters = hierarchical_clustering(paper_similarity_dict, paper_final_edges, clusters, paper_idx_2_cluster_id)

    # merge scattered papers
    clusters = merge_scattered_papers(clusters, paper_idx_2_cluster_id, title_sim_matrix, paper_all_ngbrs,
                                      paper_final_edges)

    return papers, clusters


if __name__ == '__main__':
    file_list = get_file_list('./dataset', [])

    total_papers_count = 0
    avg_pairwise_precision = 0.0
    avg_pairwise_recall = 0.0
    avg_pairwise_f1 = 0.0
    #     file_list = ['./dataset/J. Guo.xml']
    # print file_list
    for file in file_list:

        papers, clusters = name_disambiguation(file)
        total_papers_count += len(papers)

        cluster_id = 0
        for cluster_id, cluster in clusters.iteritems():
            for paper in cluster.papers:
                paper.label_predicted = cluster_id
            cluster_id += 1

        TP = 0.0  # Pairs Correctly Predicted To SameAuthor
        TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
        TP_FN = 0.0  # Total Pairs To SameAuthor

        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                if papers[i].label == papers[j].label:
                    TP_FN += 1
                if papers[i].label_predicted == papers[j].label_predicted:
                    TP_FP += 1
                if (papers[i].label == papers[j].label) \
                        and (papers[i].label_predicted == papers[j].label_predicted):
                    TP += 1
        if TP_FP == 0:
            pairwise_precision = 0
        else:
            pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        if pairwise_precision + pairwise_recall == 0:
            pairwise_f1 = 0
        else:
            pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)
        avg_pairwise_precision += pairwise_precision
        avg_pairwise_recall += pairwise_recall
        avg_pairwise_f1 += pairwise_f1

        print '\t %f' % pairwise_precision,
        print '\t %f' % pairwise_recall,
        print '\t %f' % pairwise_f1

    print len(file_list)
    print total_papers_count
    avg_pairwise_precision /= len(file_list)
    avg_pairwise_recall /= len(file_list)
    avg_pairwise_f1 /= len(file_list)
    print 'avg_pairwise_precision: %f' % avg_pairwise_precision
    print 'avg_pairwise_recall: %f' % avg_pairwise_recall
    print 'avg_pairwise_f1: %f' % avg_pairwise_f1
