import pickle
from numpy import *

def load(name):
    with open(name + '.pickle','rb') as f:
        return pickle.load(f)

def save(name, object):
    with open(name + '.pickle','wb') as f:
        pickle.dump(object, f)

def save_text(name, object):
    text = ""
    Alst = list(object)
    for ll in Alst:
        line = ", ".join( [str(l) for l in list(ll)] ) + "\n"
        text += line
    with open(name + '.txt','w') as f:
        f.write(text)

