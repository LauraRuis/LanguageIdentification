'''

File to load the codeswitching data and store it as strings
Also make sure the length is no longer than 500 characters

This python file should be in the same folder as trn-lang an trn-summary files

'''
import os 
import random
from collections import defaultdict

'''
Create dictionary with the following structure:
docid : {
    "first_lang"  : first_lang,      (The first occuring language in the file)
    "second_lang" : second_lang,     (The second occuring language in the file)
    "first_bytes" : first_bytes      (The number of bytes of the first occuring language) 
}
'''

# get the path to the file with the information
dir_path = os.path.dirname(os.path.realpath(__file__))
train_sum = os.path.join(dir_path, 'trn-summary')

# initialize dictionary
bytes_dictionary = defaultdict(lambda: dict)

# open file with the information
info_file = open(train_sum, 'r', encoding="utf-8")

# loop over file and store the information in the dictionary
for line in info_file:
    split_line = line.split(',')
    doc_id = split_line[0]
    first_lang = split_line[1]
    first_bytes = split_line[3]
    second_lang = split_line[4]
    second_bytes = split_line[5]
    bytes_dictionary[doc_id] = {
        'first_lang' : first_lang,
        'second_lang': second_lang,
        'first_bytes': first_bytes,
        'second_bytes': second_bytes
    }
info_file.close()

'''
Open each training file and convert the text inside to a small paragraph.
The files will be sliced in a way that both languages are still there. 
'''

new_train_file = os.path.join(dir_path, 'all_train.txt')
new_train_labels = os.path.join(dir_path, 'train_labels.txt')
new_train_open = open(new_train_file, 'w', encoding='utf-8')
train_label_open = open(new_train_labels, 'w', encoding='utf-8')

# get all the names of the training files
train_dir = os.path.join(dir_path, 'trn')
train_file_list = os.listdir(train_dir)

for file_id in train_file_list:
    train_file_path = os.path.join(dir_path, 'trn', file_id)
    with open(train_file_path, 'rb') as f:
        complete = bytearray()
        for line in f:
            complete.extend(line)
        
        # get the switch point from dictionary
        switch_point = int(bytes_dictionary[file_id]['first_bytes'])
        

        # get random number in range 20 - 500
        rand_number = random.randint(20, 500)

        # get characters to the left and right of switch point
        left = switch_point - rand_number
        right = switch_point + (500 - rand_number)
        
        # slice the bytearray
        sliced = complete[left:right]
        
        # The next block of code is necessary for decoding
        # During slicing it can be that a byte code is cut in two
        # Python cannot decode a file if it doesn't know the byte code, 
        # so this has to be solved
        nice_decoded = True

        try:
            decoded = sliced.decode('utf-8')
        except Exception as e:
            # The error can occur at the start or at the end of the bytearray,
            # the errormessage gives information about this
            split_error = str(e).split('position ')
            if int(split_error[1][0]) == 0:
                slice_from_start = True
            else:
                slice_from_start = False
            nice_decoded = False
            
        
        while not nice_decoded:
            if slice_from_start:
                sliced = sliced[1:]
            else:
                sliced = sliced[:-1]
            
            try: 
                decoded = sliced.decode('utf-8')
                nice_decoded = True
            except Exception as e:
                split_error = str(e).split('position ')
                if int(split_error[1][0]) == 0:
                    slice_from_start = True
                else:
                    slice_from_start = False

        # remove newlines to be able to write on one line in new file:
        no_newline = decoded.replace('\n', ' ').replace('\r', '')

        # write line to new training file
        new_train_open.write(no_newline + '\n')
        train_label_open.write(bytes_dictionary[file_id]['first_lang'] + ',' + bytes_dictionary[file_id]['second_lang'] + '\n')