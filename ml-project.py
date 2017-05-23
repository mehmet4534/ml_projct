import json

import time
import sys
import io
import os
import pickle


from numpy import loadtxt
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from io import StringIO
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from sklearn.externals import joblib

def mnbClassifier(data,lab,testdata,testlab,mnb,mnbModel):
    V=DictVectorizer();
    clf = MultinomialNB();
    if (not mnb):
        
        X = V.fit_transform(data);
        
        m=clf.fit(X,lab);
        joblib.dump(m, mnbModel)
        joblib.dump(V,"mnb-vect.sav")
    else:
        m = joblib.load(mnbModel)
        V = joblib.load("mnb-vect.sav")
        
    predict=m.predict(V.transform(testdata));
        
    sc=accuracy_score(predict,testlab);
    co = confusion_matrix(predict, testlab)
    print(co)    
        
    print(clf," MNB Classification Report \n",classification_report(predict, testlab));
    
    return True
    
  
    
def svmClassifier(data,lab2,testdata,testlab2,sv,svmModel):
    from sklearn import svm
    V=DictVectorizer();
    lin_clf = svm.LinearSVC();
    if(sv):
        lin_clf = joblib.load(svmModel) 
        V = joblib.load("svm-vect.sav")        
       
    else:
        X = V.fit_transform(data); 
        lin_clf.fit(X, lab2)
        joblib.dump(lin_clf, svmModel)
        joblib.dump(V,"svm-vect.sav")        
    
    predict=lin_clf.predict(V.transform(testdata))
    
    sc=accuracy_score(predict,testlab2);
    co = confusion_matrix(predict, testlab2)
    print(co)
      
        
    print(lin_clf," SVM Classification Report \n",classification_report(predict, testlab2));
    
    return True



file='train.txt';
filetest='test.txt';
mnbModel="mnbModel3-5.sav";
svmModel="svmModel3-5.sav";
mnb=os.path.isfile(mnbModel)
sv=os.path.isfile(svmModel)
data = list();
data2=list();
lab=list();
lab2=list();
worddict=dict();
posdict=dict();
labdict=dict();
if( not mnb or not sv):
   

    
    
    prevword='#';
    worddict.update({prevword : 1})
    prevpos='#';
    posdict.update({prevword : 1})
    w=2;
    p=2
    l=0;
    with open(file, 'r') as fobj:
        for line in fobj:
            if not line.strip():
                continue;
            (word,pos,label)=line.split(' ');
            
            
            if (word not in worddict):
                w=w+1;
                tw=w;
                worddict[word]=tw
            else:
                tw=worddict[word];
            
            if (pos not in posdict):
                p=p+1;
                tp= p;
                posdict[pos]=tp
            else:
                tp=posdict[pos];
            
            if (label not in labdict):
                l=l+1;
                tl= l;
                labdict[label]=tl
            else:
                tl=labdict[label];
            
            """if (posdict[prevpos]==None):
                tpp= ++w;
            else:"""
            tpp=posdict[prevpos];
            
            """if (worddict[prevword]==None):
                twp= ++w;
            else:"""
            twp=worddict[prevword];
            if len(data)!=0:
                data[-1]['pos+1']=pos;
                data2[-1][-1]=tp;
            d={'word' : word,'pos' : pos,'word-1' : prevword, 'pos-1': prevpos , 'pos+1': '#'};
            d2=[tw,tp,tpp,twp,0];
            prevword=word;
            prevpos=pos;
            lab.append(label);
            lab2.append(tl);
            #print(d);
            data.append(d);
            #print(d2)
            data2.append(d2);
        with open("lab2.txt", 'wb') as f:
            pickle.dump(labdict, f)
else:
    with open("lab2.txt", 'rb') as f:
        labdict = pickle.load(f)
testdata=list();
testdata2=list();
testlab=list();
testlab2=list();
prevword='#';
prevpos='#'; 

with open(filetest, 'r') as fob:
    for lin in fob:
        if lin=="\n":
            continue;
        (word,pos,label)=lin.split(' ');
        
        if(word not in worddict):
            tw=0;
        else:
            tw=worddict[word]
            
        if(prevword not in worddict):
            twp=0;
        else:
            twp=worddict[prevword]  
            
        if(pos not in posdict):
            tp=0;
        else:
            tp=posdict[pos]
            
        if(prevpos not in posdict):
            tpp=0;
        else:
            tpp=posdict[prevpos] 

        if len(testdata)!=0:
            testdata[-1]['pos+1']=pos;
            testdata[-1][-1]=tp
        d={'word' : word,'pos' : pos,'word-1' : prevword, 'pos-1': prevpos , 'pos+1': '#'};        
        
        d2=[tw,tp,twp,tpp,0];
        prevword=word;
        prevpos=pos;            
        #label={'word' : word,'pos' : pos};
        testlab.append(label);
        if ( label not in labdict):
            l=0;
        else:
            l=labdict[label]
        testlab2.append(l);
        
        testdata.append(d);
        testdata2.append(d2);

while (1==1):
    response=input("Please enter 1 for Multinomial Naive Bayes, 2 for Support Vector Machine or 3 for exit: ")
    
    if (response=="1"):
        mnb=mnbClassifier(data,lab,testdata,testlab,mnb,mnbModel)
    elif(response=="2"):
        sv=svmClassifier(data,lab2,testdata,testlab2,sv,svmModel)
    elif(response=="3"):
        break;
    else:
        print ("wrong enter")
    
 




