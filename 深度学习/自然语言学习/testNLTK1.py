#111

import nltk
from nltk.corpus import brown

#nltk.download()    # 弹出窗口让你下载  数据文件.


#brown.words()


# # 获取一段文字
# import urllib.request
# response = urllib.request.urlopen('http://php.net/')
# html = response.read()
# print(html)


# #计算每一个词的频度
# from bs4 import BeautifulSoup
# import urllib.request
# import nltk
# response = urllib.request.urlopen('http://php.net/')
# html = response.read()
# soup = BeautifulSoup(html,"html5lib")
# text = soup.get_text(strip=True)
# tokens = [t for t in text.split()]
# freq = nltk.FreqDist(tokens)
# for key,val in freq.items():
#     print (str(key) + ':' + str(val))


#显示英语中有哪些  停止词(就是不重要的,影响阅读的吃,  如 .the ,a ,of, an,a)

from nltk.corpus import stopwords
stopwords.words('english')