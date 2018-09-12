import re
import requests
from lxml import etree
import os

def get_download_url(url):
    with requests.get(url) as f:
        data = f.text
        html = etree.HTML(data)
       
    links = html.xpath('//dd[@class="a1"]/text()')
    return links

for i in range(1,100):
    with requests.get('http://www.zhuixinfan.com/viewtvplay-' + str(i) + '.html') as f:
        data = f.text
        html = etree.HTML(data)

    dramaname = re.search(r'<span id=\"pdtname\">(.+?)</span>', data)   
    if dramaname != None:
        dramaname = dramaname.group(1)
    else:
        continue
    dramaname = '## ' + dramaname + '  \n'
    with open('D:\学习\毕业设计\crawel\drama.txt', 'a+') as f:
            f.write(dramaname)

    url_1 = html.xpath('//td[@class="td2"]')
    for item in url_1:
        href = item.xpath('./a/@href')[0]
        title = item.xpath('./a/text()')[0]
        download_links = get_download_url('http://www.zhuixinfan.com/' + href)
        with open('D:\学习\毕业设计\crawel\drama.txt', 'a+') as f:
            f.write("### "+title+'\n')
            f.write("### "+"电驴："+str(download_links[0])+'\n')
            f.write("### "+"Torrent："+str(download_links[1])+'\n')

print('ok')