#  %

# %


import os
import sys
import activelearning as al
from sys import argv
import shutil



if __name__ == '__main__':
    
    a = al.sampling2(argv[1])
    print(a)
    
    for i in al.copyfiles():
        
        shutil.copyfile(argv[1]+i, 'selected/' + i)
        
        #pathss = i.split('/')[-1]
        #shutil.copyfile(i, 'selected/' + pathss)
    