import requests
from bs4 import BeautifulSoup
import jieba
from wordcloud import WordCloud
import numpy as np
from PIL import Image

# 将豆瓣电影评论URL地址，赋值给变量url
url = "https://movie.douban.com/subject/10463953/comments?status=P"

# 将User-Agent以字典键对形式赋值给headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}

# 将字典headers传递给headers参数，添加进requests.get()中，赋值给response
response = requests.get(url, headers=headers)

# 使用.text属性获取网页内容，并赋值给html
html = response.text

# 使用BeautifulSoup()传入变量html和解析器lxml，赋值给soup
soup = BeautifulSoup(html, "lxml")

# 使用find_all()查询soup中class="short"的节点，赋值给content_all
content_all = soup.find_all(class_="short")

# 排除词
excludes = {'span', 'class', 'short', 'not'}

# 进行词汇整理
content = str(content_all)
words = jieba.cut(content)
text = ""
for i in words:
    if len(i) > 1:
        text += " "
        text += i

# 打开图片文件赋值mask
mask = np.array(Image.open("Apple.jpg"))

# 创建WordCloud对象，赋值给wordCloud
wordCloud = WordCloud(background_color="white", repeat=False, max_words=100, max_font_size=300, colormap="Blues",
                      font_path="Fonts/STXINGKA.TTF", mask=mask, stopwords=excludes)

# 向WordCloud对象中加载文本text
wordCloud.generate(text)

# 将词云图输出为图像文件
wordCloud.to_file("The Imitation Game.png")

# 使用print输出 success
print("success")
