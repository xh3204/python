import MySQLdb
from datetime import datetime

def author2venue(authorID):
    conn = MySQLdb.connect(host='202.120.36.29', port=3306, user='groupleader', passwd='onlyleaders', db='mag-new-160205',
                            charset="utf8")
    cursor = conn.cursor()
    author2paper = 'SELECT PaperID FROM PaperAuthorAffiliations WHERE AuthorID="%s"'
    paper2venue_str = 'SELECT NormalizedVenueName FROM Papers WHERE PaperID="%s"'
    paper2scicitation = 'SELECT SCICitation FROM PaperSciReferencesCount WHERE PaperReferenceID="%s"'
    paper2citation = 'SELECT CitationCount FROM PaperCitationCount WHERE PaperID="%s"'
    cursor.execute(author2paper % authorID)
    paperID = cursor.fetchall()
    paperID = [paper[0] for paper in paperID]
    
    # map the paper to the number of the conference
    paper2conf = {}
    for paper in paperID:
        cursor.execute(paper2citation % paper)
        citation = cursor.fetchall()
        if citation != tuple():
            paper2conf[paper] = citation[0][0]
        else:
            paper2conf[paper] = None

    # map the paper to the venue, and calculate the paper number of each venue
    venues = {}
    paper2venue = {}
    for paper in paperID:
        cursor.execute(paper2venue_str % paper)
        venue = cursor.fetchall()
        paper2venue[paper] = venue[0][0]
        
        if paper2venue[paper] not in venues.keys():
            if paper2conf[paper] != None:
                venues[paper2venue[paper]] = paper2conf[paper]
        else:
            if paper2conf[paper] != None:
                venues[paper2venue[paper]] += paper2conf[paper]
    
    return venues

venues = author2venue('7EA9B3E8')
print(venues)