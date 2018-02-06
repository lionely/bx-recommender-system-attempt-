#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:27:38 2018

@author: NewType
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:15:41 2018

@author: NewType
"""

#Questions:
# Does it make sense intuitvely?
# If it does, to validate should I just do k-fold cross validation
# what else can I do?

#Book recommendation system using AutoEncoders

#Importing the libraries
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Resources
#https://rachellegardner.com/isbn-10-isbn-13-and-those-pesky-x%E2%80%99s/
#https://sellercentral.amazon.com/forums/thread.jspa?threadID=17640
#Resources

# Importing the dataset
#books = pd.read_csv('BX-CSV-Dump/BX-Books.csv', sep = ';', header = None,engine= 'python',encoding = 'latin-1',error_bad_lines=False)
#users = pd.read_csv('BX-CSV-Dump/BX-Users.csv', sep = ';', header = None,engine= 'python',encoding = 'latin-1',error_bad_lines=False)
n = 1149767 #number of records in file
s = 10000 #desired sample size
skip = sorted(random.sample(xrange(n),n-s))
ratings = pd.read_csv('BX-CSV-Dump/BX-Book-Ratings.csv', sep = ';', header = None,engine= 'python',encoding = 'latin-1',error_bad_lines=False,skiprows=skip)
#TODO removing those pesky X's as they mess with my mapping. just map them actually.
#There are also entries that contain letters. These need to be dealt with.

#TODO, remap USER IDs, so it is easy to get nb_users
id_users = ratings[0].unique()
user_map = {}
user_count = 0
for user in id_users:
    user_map[user] = user_count
    user_count+=1

id_books = ratings[1].unique()
books_map = {}
book_count = 0
for book in id_books:
    books_map[book] = book_count
    book_count+=1
    
ratings_user_mapped = ratings[0].map(user_map)
ratings_book_mapped = ratings[1].map(books_map)
pieces = [ratings_user_mapped, ratings_book_mapped ,ratings[2]]
mapped_ratings = pd.concat(pieces,axis=1)


# Preparing the training set and the test set
#TODO need to train-test split into batches training set is way too large.
training_set, test_set = train_test_split(mapped_ratings,test_size=0.20)#changed from False to True

training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_books = int(max(max(training_set[:,1]),max(test_set[:,1])))
#nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
# pyTorch needs arrays as inputs.
#t_set = training_set[:,1][training_set[:,0] == 3 ]

def convert(data): #check
    new_data = []
    for id_user in range(0, nb_users):
        id_book = data[:,1][data[:,0] == id_user]
        id_ratings = data[:,2][data[:,0] == id_user]
        ratings = np.zeros(nb_books)
        #id_movies-1 substracts 1 from each element in np array
        ratings[id_book] = id_ratings #numpy matches things up.
        #print("id_book is: ",id_book , "id_user is: ",id_user)
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
# tensors are mutlidimensional arrays that contain elements of a single data type.
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self):
        super(SAE,self).__init__()
        self.fc1 = nn.Linear(nb_books, 20)#with 20 we trying to detect 20 features
        self.fc2 = nn.Linear(20, 10)
        #self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(10, 20 )#reconstructing matches first encode
        #self.dropout_2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(20, nb_books )#reconstructed features
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        #x = self.dropout(x)
        x = self.activation(self.fc3(x))
        #x = self.dropout_2(x)
        x = self.fc4(x) #vector of predicted ratings
        return x

parameters = {'lr':[0.01],
              'nb_epoch': [5],
              'optimizer':['adam','rmsprop'],
              'weight_decay':[0.5]}
# for each lr do each hyper param.
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # decay decrease learning rate every few epochs
#def gridSearch(sae):
    
# Training the SAE
nb_epoch = 100 #200 causes overfitting, dropout did not help.
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)#pytorch needs this dummy dimension
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False # don't compare against target
            output[target == 0] = 0
            loss = criterion(output,target)
            mean_corrector = nb_books/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector) #this has the loss in the loss object
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch)+' loss: '+str(train_loss/s))
    

# Testing the SAE
test_loss = 0
s = 0.
output_numpy_all = np.empty([0,nb_books])
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)#pytorch needs this dummy dimension
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False # don't compare against target
        output[target == 0] = 0
        output_numpy = output.data.numpy()
        output_numpy_all = np.vstack((output_numpy_all, output_numpy) )
        loss = criterion(output,target)
        mean_corrector = nb_books/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector) #this has the loss in the loss object
        s += 1.
print('Test loss is: '+str(test_loss/s))

#saving model parameters
def save(model,optimizer):
    torch.save({'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
               }, 'last_model.pth')

   
def load(model,optimizer):
    if os.path.isfile('last_model.pth'):
        print("=> loading checkpoint... ")
        checkpoint = torch.load('last_model.pth')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("done !")
    else:
        print("no checkpoint found...")


save(sae,optimizer)      

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_