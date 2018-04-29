#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:40:31 2017
Change Log
@author: shubham
"""
print("========="*3,'LOGS',3*"========")
def show_logs(log):
    print('=='*30)
    for items in log:
        print(items)
    print('--'*30)
logs1=['DATE:19 JUN 2017  CP: base']
logs1.append('1. modified split_input_sgd.py to be compatible with edge loss.')
logs1.append('2. reduced the number of filters in it to reduce the number of params.')
logs1.append('3. test_net uses custom_loss_functions instead of test_custom_loss_functions.')
logs1.append('\n\nDATE:20 JUN 2017  CP:12')
logs1.append('1.Created working proto-type for Tester.')
logs1.append('DATE:22 JUN 2017  CP: 14')
logs1.append('Added warnings in data.py and modified readFilesAsNdArray().')
logs1.append('Losses are heavily modified. Consistency test also added.')
logs1.append('Note: Make sure you use the right version of losses and models.')
logs1.append('')
logs1.append('')
logs1.append('')
logs1.append('')

show_logs(logs1)


#%%NOTES:
notes=[]
notes.append('1. train_with_gen takes care of input channel/branches using generator and not by itself.')
notes.append('2. Models use default params in default.py . Be sure to change dimensions in default before tesing on \n'+
             'data of size other than that mentioned in the default.')
notes.append('')
notes.append('')
notes.append('')

show_logs(notes)

