##Setting Up Environment for Project
1. copy ml_project folder to ant folder on your computer, preferably your virtualenv folder
2. activate virtualenv
    '''
    Mac OS : source ml_project/env/bin/activate 
    Windows :
            PS C:\> Set-ExecutionPolicy AllSigned  #When using PowerShell     
            \path\to\ml_project\Scripts\activate

    '''
##Deactivate environment
    '''
    deactivate
    '''
#####solve matplotlib: RuntimeError: Python is not installed as a framework
'''bash
echo 'backend: TkAgg' >> ~/.matplotlib/matplotlibrc

OR

import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  

<<<<<<< HEAD:evaluate/README.md
This repository houses the cleaning of the data for a text-classfication problem. 

### How do I get set up? ###
=======
'''
>>>>>>> origin/env:README.md


<<<<<<< HEAD:evaluate/README.md
Download the file. 
Ensure there is an mbti_1.csv file in the same directory as clean.py
Open a terminal and navigate to the directory containing clean.py
run clean.py: python clean.py

The process will automatically generate the types.txt file and a cleaned_data.txt file to the root directory

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact


##merging all necessary repos
=======

>>>>>>> origin/env:README.md
