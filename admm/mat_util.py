import pickle

def load(name):
    with open(name + '.pickle','rb') as f:
        return pickle.load(f)

def save(name, object) :
    with open(name + '.pickle','wb') as f:
        pickle.dump(object, f)
