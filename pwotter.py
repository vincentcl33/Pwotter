
# language
from operator import index
from cv2 import cvtColor
from nltk.corpus import movie_reviews, stopwords
from nltk.classify import NaiveBayesClassifier, accuracy
from string import punctuation
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

# Computer ViSION
import random, pickle
import cv2
from matplotlib import pyplot as plt

# web scrapping
import requests
from bs4 import BeautifulSoup
import numpy as np

# Methods
def preprocess(words):
    # remove stopwords
    en_stopwords = stopwords.words('english')
    clean_words = []
    for word in words:
        if word not in en_stopwords:
            clean_words.append(word)

    # remove punctuation
    clean_punctuation = []
    for word in clean_words:
        if word not in punctuation:
            clean_punctuation.append(word)

    # lowercase
   
   
    return clean_punctuation 

    clean_punctuation.lowercase()


def train():
    document = []
    all_words = []
    for file_id in movie_reviews.fileids():
        for category in movie_reviews.categories(file_id):
            words = movie_reviews.words(file_id)
            # preprocess dataset
            words = preprocess(words)
            # get all words
            all_words += words
            document.append((words, category))

    all_words = list(FreqDist(all_words).keys())
    all_words = all_words[:500]

    #dataset feature
    df = []
    for words, category in document:
        feature = {}
        for word in all_words:
            feature[word] = (word in words)
        
        df.append((feature, category))

    # training
    random.shuffle(df)
    count = int (len(df) * 0.9)

    train_df = df[:count]
    test_df = df[count:]

    classifier = NaiveBayesClassifier.train(train_df)
    acc = accuracy(classifier, test_df) * 100

    print('Accuracy : {}%', format(acc))

    # save model
    file = open('model.pickle','wb')
    pickle.dump(classifier, file)
    file.close()

    return classifier
# load model
try:
    file = open('model.pickle', 'rb')
    classifier = pickle.load(file)
    file.close()
except:
    classifier = train()

def equ_hist_image(image):
    # Grayscaling Image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize image
    equ_img = cv2.equalizeHist(img_gray)

    # show image and plot
    plt.subplot(1, 2, 1)
    plt.hist(img_gray.flat, bins=256, range=(0,255), histtype='step')
    plt.xlabel("Intensify Value")
    plt.ylabel('Intensify Quantity')

    plt.subplot(1, 2, 2)
    plt.hist(equ_img.flat, bins=256, range=(0,255), color= "red", histtype='step')
    plt.xlabel("Intensify Value")
    plt.ylabel('Intensify Quantity')


    plt.show()

def equ_image(image):
    # Grayscaling Image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize image
    equ_img = cv2.equalizeHist(img_gray)
    
    # show image and plot
    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(equ_img, cmap='gray')


    plt.show()


#web scrap
url = 'https://academicslc.github.io/E222-COMP6683-YT01-00/'
html_page = requests.get(url)
page_content = html_page.content

soup = BeautifulSoup(page_content, 'html.parser')

paragraphs = soup.find_all('div', class_="user-post-content" )

for paragraph in paragraphs:
    text = paragraph.get_text()
    words = word_tokenize(text)
    classify_result = classifier.classify(FreqDist(words))
 

    print("Post : {}". format(classify_result))

#computer vision 
images = soup.find_all('img', class_='rounded-circle user-image')
for image in images:
    img_src = url + image.attrs['src']
    
    r = requests.get(img_src)
    image = np.asarray(bytearray(r.content), dtype= 'uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    equ_hist_image(image)
    equ_image(image)


# images = soup.find("img", attrs={"class":"rounded-circle user-image", "src":"images/dwayne_johnson.jpeg"})
# # for image in images:
# #     img_src = image.attrs[]
    
# r = requests.get(url + images)
# image = np.asarray(bytearray(r.content), dtype= 'uint8')
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
# equ_hist_image(image)
# equ_image(image)
