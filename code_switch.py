'''

File to load the codeswitching data and store it as strings
Also make sure the length is no longer than 500 characters

This python file should be in the same folder as trn-lang an trn-summary
(In the altw2010-langid folder)

'''
import argparse
import os 
import random
from collections import defaultdict

def bytes_to_string(text_piece):

    nice_decoded = True

    try:
        decoded = text_piece.decode('utf-8')
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
            text_piece = text_piece[1:]
        else:
            text_piece = text_piece[:-1]
        
        try: 
            decoded = text_piece.decode('utf-8')
            nice_decoded = True
        except Exception as e:
            split_error = str(e).split('position ')
            if int(split_error[1][0]) == 0:
                slice_from_start = True
            else:
                slice_from_start = False

    # remove newlines to be able to write on one line in new file:
    no_newline = decoded.replace('\n', ' ').replace('\r', '')

    return no_newline

def create_dict(name_dataset):
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
    summary_file = os.path.join(dir_path, name_dataset + '-summary')

    # initialize dictionary
    bytes_dictionary = defaultdict(lambda: dict)

    # open file with the information
    info_file = open(summary_file, 'r', encoding="utf-8")

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
    return bytes_dictionary

def write_to_files(dictionary, name_dataset):
    '''
    Open each file in the given dataset and convert the text inside to a small paragraph.
    The files will be sliced in a way that both languages are still there. 
    '''
    # get the path of the directory to use later
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # initialize filenames of files to store to
    sentences_file = os.path.join(dir_path, name_dataset + '_sentences.txt')
    lang_labels_file = os.path.join(dir_path, name_dataset + '_lang_labels.txt')
    switch_labels_file = os.path.join(dir_path, name_dataset + '_switch_labels.txt')

    # open the filenames so it's possible to write to them
    sentences_open = open(sentences_file, 'w', encoding='utf-8')
    lang_labels_open = open(lang_labels_file, 'w', encoding='utf-8')
    switch_labels_open = open(switch_labels_file, 'w', encoding='utf-8')

    # get all the names of the document files
    docs_dir = os.path.join(dir_path, name_dataset)
    file_list = os.listdir(docs_dir)

    for file_id in file_list:
        file_path = os.path.join(dir_path, name_dataset, file_id)
        with open(file_path, 'rb') as f:
            complete = bytearray()
            for line in f:
                complete.extend(line)
            
            # get the switch point from dictionary
            switch_point = int(dictionary[file_id]['first_bytes'])
            
            # get random number in range 20 - 480
            rand_number = random.randint(20, 480)

            # get characters to the left and right of switch point
            left = switch_point - rand_number
            right = switch_point + (500 - rand_number)

            # for small reference only need right point
            small_right = switch_point + 15

            # slice the bytearray
            sliced = complete[left:right]
            small = complete[switch_point:small_right]

            sliced_clean = bytes_to_string(sliced)
            small_clean = bytes_to_string(small)

            new_switch_point = sliced_clean.find(small_clean)

            # write line to new training file
            sentences_open.write(sliced_clean + '\n')
            lang_labels_open.write(dictionary[file_id]['first_lang'] + ',' + dictionary[file_id]['second_lang'] + '\n')
            switch_labels_open.write(str(new_switch_point) + '\n')
    
    sentences_open.close()
    lang_labels_open.close()
    switch_labels_open.close()


def main():
    if FLAGS.dataset == 'train':
        set_name = 'trn'
    elif FLAGS.dataset == 'valid':
        set_name = 'dev'
    elif FLAGS.dataset == 'test':
        set_name = 'tst'

    information_dictonary = create_dict(set_name)

    write_to_files(information_dictonary, set_name)



if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'train',
                        help='name of dataset, can either be train, valid or test')
    FLAGS, unparsed = parser.parse_known_args()

    main()