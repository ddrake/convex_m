import pickle
from numpy import *

def load(name, directory=None):
    path = directory + '/' + name if directory else name
    with open(path + '.pickle','rb') as f:
        return pickle.load(f)

def save(name, object, directory=None):
    path = directory + '/' + name if directory else name
    with open(path + '.pickle','wb') as f:
        pickle.dump(object, f)

def save_text(name, object):
    text = ""
    Alst = list(object)
    for ll in Alst:
        line = ", ".join( [str(l) for l in list(ll)] ) + "\n"
        text += line
    with open(name + '.txt','w') as f:
        f.write(text)

