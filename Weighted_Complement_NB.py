import pandas as pd 
import numpy as np 
from collections import defaultdict
import re

def preprocess_string(str_arg):
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case
    cleaned_str=cleaned_str.strip()
    return cleaned_str # returning the preprocessed string


class CompNaiveBayes:
    
    def __init__(self,unique_classes):
        
        self.classes=unique_classes # Constructor is sinply passed with unique number of classes of the training set
        

    def addToBow(self,example,dict_index):
        
        if isinstance(example,np.ndarray): example=example[0]
     
        for token_word in example.split(): #for every word in preprocessed example
          
            self.bow_dicts[dict_index][token_word]+=1 #increment in its count
            
    def train(self,dataset,labels):
        self.examples=dataset
        self.labels=labels
        self.bow_dicts=np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])
        
        #only convert to numpy arrays if initially not passed as numpy arrays - else its a useless recomputation
        
        if not isinstance(self.examples,np.ndarray): self.examples=np.array(self.examples)
        if not isinstance(self.labels,np.ndarray): self.labels=np.array(self.labels)
            
        #constructing BoW for each category
        for cat_index,cat in enumerate(self.classes):
          
            all_cat_examples=self.examples[self.labels==cat] #filter all examples of category == cat
            
            #get examples preprocessed
            
            cleaned_examples=[preprocess_string(cat_example) for cat_example in all_cat_examples]
            
            cleaned_examples=pd.DataFrame(data=cleaned_examples)
            
            #now costruct BoW of this particular category
            np.apply_along_axis(self.addToBow,1,cleaned_examples,cat_index)
            
        prob_classes=np.empty(self.classes.shape[0])
        all_words=[]
        cat_word_counts=np.empty(self.classes.shape[0])
        total=0
        for cat_index,cat in enumerate(self.classes):
           
            #Calculating prior probability p(c) for each class
            prob_classes[cat_index]=np.sum(self.labels==cat)/float(self.labels.shape[0]) 
            
            #Calculating total counts of all the words of each class 
            count=list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index]=np.sum(np.array(list(self.bow_dicts[cat_index].values())))+1 # |v| is remaining to be added
            total+=cat_word_counts[cat_index]
            #get all words of this category                                
            all_words+=self.bow_dicts[cat_index].keys()
                                                     
        
        #combine all words of every category & make them unique to get vocabulary -V- of entire training set
        
        self.vocab=np.unique(np.array(all_words))
        self.vocab_length=self.vocab.shape[0]
                                  
        #computing denominator value                                      
        denoms=np.array([total-cat_word_counts[cat_index] for cat_index,cat in enumerate(self.classes)])                                                                          
#         print(len(dataset))
#         print(cat_word_counts[cat_index])
        self.cats_info=[(self.bow_dicts[cat_index],prob_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(self.classes)]                               
        self.cats_info=np.array(self.cats_info)                                 
        self.wc_count=np.ones((self.classes.shape[0],self.vocab.shape[0]))
        for cl_ind,cl in enumerate(self.classes):
            cl_count=1
            for i,w in enumerate(self.vocab):
                count=1
                for comp_cl_ind,comp_cl in enumerate(self.classes):
                    if comp_cl!=cl:
                        count+=self.cats_info[comp_cl_ind][0].get(w,0)
                self.wc_count[cl_ind][i]=np.log(count/float(denoms[cl_ind]))
                cl_count+=abs(np.log(count/float(denoms[cl_ind])))
            self.wc_count[cl_ind]=self.wc_count[cl_ind]/float(cl_count)
                                              
    def getExampleProb(self,test_example):
        
        likelihood_prob=np.zeros(self.classes.shape[0]) #to store probability w.r.t each class
        
        #finding probability w.r.t each class of the given test example
        for cat_index,cat in enumerate(self.classes): 
                             
            for test_token in test_example.split(): #split the test example and get p of each test word
                #This loop computes : for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ]   
                #get total count of this test token from it's respective training dict to get numerator value
#                 test_token_counts=1
#                 for cat_index1,cat1 in enumerate(self.classes):
#                     if cat!=cat1:
#                         test_token_counts+=self.cats_info[cat_index1][0].get(test_token,0)
#                 print(self.cats_info[cat_index][2])
#                 print(type(self.cats_info[cat_index][2]))
                #now get likelihood of this test_token word                              
#                 test_token_prob=test_token_counts/float(self.cats_info[cat_index][2])                              
                
                #remember why taking log? To prevent underflow!
#                 print(test_token_prob)
                
                w_ind= np.where(self.vocab==test_token)[0]
                if len(w_ind)>0:
                    likelihood_prob[cat_index]+=self.wc_count[cat_index][int(w_ind)]
#             print()
        
        # we have likelihood estimate of the given example against every class but we need posterior probility
        post_prob=np.empty(self.classes.shape[0])
        for cat_index,cat in enumerate(self.classes):
            post_prob[cat_index]=-likelihood_prob[cat_index]                                  
#         print(post_prob)
        return post_prob
    
   
    def test(self,test_set):
        
        predictions=[] #to store prediction of each test example
        for example in test_set: 
            print(i)                       
            #preprocess the test example the same way we did for training set exampels                                  
            cleaned_example=preprocess_string(example) 
             
            #simply get the posterior probability of every example                                  
            post_prob=self.getExampleProb(cleaned_example) #get prob of this example for both classes
        
            #simply pick the max value and map against self.classes!
            predictions.append(self.classes[np.argmax(post_prob)])
        
        return np.array(predictions)



from sklearn.datasets import fetch_20newsgroups
# categories=['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med'] 
newsgroups_train=fetch_20newsgroups(subset='train')
train_data=newsgroups_train.data #getting all trainign examples
train_labels=newsgroups_train.target #getting training labels


newsgroups_test=fetch_20newsgroups(subset='test') #loading test data
test_data=newsgroups_test.data #get test set examples
test_labels=newsgroups_test.target #get test set labels


cnb=CompNaiveBayes(np.unique(train_labels)) #instantiate a NB class object
cnb.train(train_data,train_labels) #start tarining by calling the train function
pclasses=cnb.test(test_data)
test_acc=np.sum(pclasses==test_labels)/float(test_labels.shape[0]) 

print ("Test Set Examples: ",test_labels.shape[0]) # Outputs : Test Set Examples:  1502
print ("Test Set Accuracy: ",test_acc*100,"%")