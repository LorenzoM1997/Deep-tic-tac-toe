import os
import pickle

def save_mct(mct):
    mct_filename = "mct.txt"
    with open(mct_filename, "wb") as fp:   #Pickling
        pickle.dump(mct, fp)
    print("Monte Carlo Tree saved correctly")

def load_mct():
    mct_filename = "mct.txt"
    if os.path.isfile(mct_filename):
        # load existing file
        print("Found existing Monte Carlo Tree file. Opening it")
        with open(mct_filename, "rb") as fp:   # Unpickling
            mct = pickle.load(fp)
    else:
        print("Creating new Monte Carlo Tree.")
        mct = []
    return mct
