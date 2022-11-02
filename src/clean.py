# Convert email into feature vector
# Create Test & Training Set
# Add Hyperparameters to:
# - Strip email headers
# - Convert to lowercase
# - Remove punctuation
# - Replace urls with "URL"
# - Replace numbers with "NUMBER"
# - Perform Stemming (trim word endings with library)

import pandas as pd
import numpy as np
import os
import email
import email.policy
from bs4 import BeautifulSoup

os.listdir('../input/hamnspam/')

ham_filenames = [name for name in sorted(os.listdir('../input/hamnspam/ham')) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir('../input/hamnspam/spam')) if len(name) > 20]

print('Amount of ham files:', len(ham_filenames))
print('Amount of spam files:', len(spam_filenames))    
print('Spam to Ham Ratio:',len(spam_filenames)/len(ham_filenames))