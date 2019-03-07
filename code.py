# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#E查价钱
from iexfinance.stocks import Stock

def find_price(name):
    stock_price = Stock(name)
    appe=stock_price.get_price()
    return appe



#F查交易量
from iexfinance.stocks import Stock

def find_volume(name):
    stock_price = Stock(name)
    appe=stock_price.get_volume()
    return appe



#D实体抽取
import spacy
import numpy as np
sentences = [' i want to fly from boston at 838 am and arrive in denver at 1110 in the morning', 
 ' what flights are available from pittsburgh to baltimore on thursday morning', 
 ' what is the arrival time in san francisco for the 755 am flight leaving washington', 
 ' cheapest airfare from tacoma to orlando', 
 ' round trip fares from pittsburgh to philadelphia under 1000 dollars', 
 ' i need a flight tomorrow from columbus to minneapolis', 
 ' what kind of aircraft is used on a flight from cleveland to dallas', 
 ' show me the flights from pittsburgh to los angeles on thursday', 
 ' all flights from boston to washington', 
 ' what kind of ground transportation is available in denver', 
 ' show me the flights from dallas to san francisco', 
 ' show me the flights from san diego to newark by way of houston', 
 ' what is the cheapest flight from boston to bwi', 
 ' all flights to baltimore after 6 pm', 
 ' show me the first class fares from boston to denver', 
 ' show me the ground transportation in denver', 
 ' all flights from denver to pittsburgh leaving after 6 pm and before 7 pm', 
 ' i need information on flights for tuesday leaving baltimore for dallas dallas to boston and boston to baltimore', 
 ' please give me the flights from boston to pittsburgh on thursday of next week', 
 ' i would like to fly from denver to pittsburgh on united airlines', 
 ' show me the flights from san diego to newark', 
 ' please list all first class flights on united from denver to baltimore', 
 ' what kinds of planes are used by american airlines', 
 " i'd like to have some information on a ticket from denver to pittsburgh and atlanta", 
 " i'd like to book a flight from atlanta to denver", 
 ' which airline serves denver pittsburgh and atlanta', 
 " show me all flights from boston to pittsburgh on wednesday of next week which leave boston after 2 o'clock pm", 
 ' atlanta ground transportation', ' i also need service from dallas to boston arriving by noon', 
 ' show me the cheapest round trip fare from baltimore to dallas']


# Load the spacy model: nlp, en_core_web_md
nlp = spacy.load('en_core_web_md')

# Calculate the length of sentences
n_sentences = len(sentences)

# Calculate the dimensionality of nlp
embedding_dim = nlp.vocab.vectors_length

print(n_sentences)
print(embedding_dim)

# Initialize the array with zeros: X
X = np.zeros((n_sentences, embedding_dim))

# Iterate over the sentences
for idx, sentence in enumerate(sentences):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X[idx, :] = doc.vector

    
import pandas as pd
X_train = pd.read_csv('SVM/X_train.csv')
X_test = pd.read_csv('SVM/X_test.csv')
y_train = pd.read_csv('SVM/y_train.csv')['label']
y_test = pd.read_csv('SVM/y_test.csv')['label']


# Import SVC
from sklearn.svm import SVC

# Create a support vector classifier
clf = SVC()

# Fit the classifier using the training data
clf.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Count the number of correct predictions
n_correct = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        n_correct += 1
        
print("Predicted {0} correctly out of {1} test examples.".format(n_correct, len(y_test)))


include_entities = [ 'ORG']

def find_company(message):
    ents = dict.fromkeys(include_entities)
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ in include_entities:
            # Save interesting entities
            ents[ent.label_] = ent.text
    return ents['ORG']




#C意图识别
# Import necessary modules
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

# Create a trainer that uses this config
trainer = Trainer(config.load("config_spacy.yml"))

# Load the training data
training_data = load_data('xtc-rasa-fif.json')

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

responses = 'the price of {1} is {2}'

def find_intent(message):
    # Extract the entities
    intent = interpreter.parse(message)['intent']['name']
    return intent




#B回复函数
pending=None
company=None

def respond(message):
    global pending
    global company
    response=None
    intent=find_intent(message)
    if pending=='pending_target':
        if intent=='price_search':
            response='the price of {} is {}'.format(company,find_price(company))
            pending=None
            company=None
            return response
        elif intent=='volume_search':
            response='the volume of {} is {}'.format(company,find_volume(company))
            pending=None
            company=None
            return response
        else:
            response='What imformation do you want to know, price or volume?'
            return response
    else:
        company=find_company(message)
        if company is not None:
            if intent=='price_search':
                response='the price of {} is {}'.format(company,find_price(company))
                company=None
                return response
            elif intent=='volume_search':
                response='the volume of {} is {}'.format(company,find_volume(company))
                company=None
                return response
            else:
                pending='pending_target'
                response='What imformation do you want to know, price or volume?'
                return response
        else:
            if intent=='greet':
                response='Hi,can I help you?'
                return response
            elif intent=='thank':
                response='you are welcome'
                return response
            elif intent=='ask_function':
                response='I can help you search the price and volume of a stock,which stock do you want to know.Please tell me the stock code'
                return response
            else:
                response='hello,I am a stock inquiry robot'
                return response
            
            
            
#A微信回复函数
from wxpy import *

bot = Bot()

my_friend = bot.friends().search('xtc')[0]
boring_group = bot.groups().search('test')[0]

@bot.register([my_friend, Group], TEXT)
def auto_reply(msg):
    print(msg.text)
    # 如果是群聊，但没有被 @，则不回复
    if isinstance(msg.chat, Group) and not msg.is_at:
        return '0'
    else:
        # 回复消息内容和类型
        return respond(msg.text)