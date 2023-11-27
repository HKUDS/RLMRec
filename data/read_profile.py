import pickle
import numpy as np

with open('data/amazon/usr_prf.pkl', 'rb') as f:
    prf = pickle.load(f)

user_index = np.random.choice(a=len(prf), size=1)[0]

print("User {}'s Profile:\n".format(user_index))
print("PROFILE: {}\n".format(prf[user_index]['profile']))
print("REASONING: {}".format(prf[user_index]['reasoning']))