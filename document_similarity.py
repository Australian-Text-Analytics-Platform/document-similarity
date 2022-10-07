#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:09:13 2022

@author: sjuf9909
"""
# import required packages
import codecs
import hashlib
import io
import os
import zipfile
from tqdm import tqdm
from zipfile import ZipFile
from pathlib import Path
from itertools import chain

# import tools to calculate Jaccard similarity
from datasketch import MinHash, MinHashLSH

# pandas: tools for data processing
import pandas as pd

# seaborn: visualization tools
import seaborn as sns

# html visualization
from diffviz import html_diffs

# NLTK and gensim: natural language processing tools for working with language/text data
import nltk
from nltk import ngrams
from gensim.utils import tokenize

# ipywidgets: tools for interactive browser controls in Jupyter notebooks
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display, clear_output, FileLink, HTML

# import other packages
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()


class DownloadFileLink(FileLink):
    '''
    Create link to download files in Jupyter Notebook
    '''
    html_link_str = "<a href='{link}' download={file_name}>{link_text}</a>"

    def __init__(self, path, file_name=None, link_text=None, *args, **kwargs):
        super(DownloadFileLink, self).__init__(path, *args, **kwargs)

        self.file_name = file_name or os.path.split(path)[1]
        self.link_text = link_text or self.file_name

    def _format_path(self):
        from html import escape

        fp = "".join([self.url_prefix, escape(self.path)])
        return "".join(
            [
                self.result_html_prefix,
                self.html_link_str.format(
                    link=fp, file_name=self.file_name, link_text=self.link_text
                ),
                self.result_html_suffix,
            ]
        )
        

class DocumentSimilarity():
    '''
    Using Jaccard similarity to identify similar documents in a corpus
    '''
    def __init__(self):
        '''
        Initiate the DocumentSimilarity
        '''
        # initiate other necessary variables
        self.large_file_size = 1000000
        
        # create an output folder if not already exist
        os.makedirs('output', exist_ok=True)
        
        # initiate the variables for file uploading
        self.file_uploader = widgets.FileUpload(
            description='Upload your files (txt, csv, xlsx or zip)',
            accept='.txt, .xlsx, .csv, .zip', # accepted file extension
            multiple=True,  # True to accept multiple files
            error='File upload unsuccessful. Please try again!',
            layout = widgets.Layout(width='320px')
            )
    
        self.upload_out = widgets.Output()
        
        # give notification when file is uploaded
        def _cb(change):
            with self.upload_out:
                # clear output and give notification that file is being uploaded
                clear_output()
                
                # check file size
                self.check_file_size(self.file_uploader)
                
                # reading uploaded files
                self.process_upload()
                
                # clear saved value in cache and reset counter
                self.file_uploader._counter=0
                self.file_uploader.value.clear()
                
                # give notification when uploading is finished
                print('Finished uploading files.')
                print('{} text documents are loaded for tagging.'.format(self.text_df.shape[0]))
            
        # observe when file is uploaded and display output
        self.file_uploader.observe(_cb, names='data')
        self.upload_box = widgets.VBox([self.file_uploader, self.upload_out])
        
        # CSS styling 
        self.style = """
        <style scoped>
            .dataframe-div {
              max-height: 250px;
              overflow: auto;
              position: relative;
            }
        
            .dataframe thead th {
              position: -webkit-sticky; /* for Safari */
              position: sticky;
              top: 0;
              background: #2ca25f;
              color: white;
            }
        
            .dataframe thead th:first-child {
              left: 0;
              z-index: 1;
            }
        
            .dataframe tbody tr th:only-of-type {
                    vertical-align: middle;
                }
        
            .dataframe tbody tr th {
              position: -webkit-sticky; /* for Safari */
              position: sticky;
              left: 0;
              background: #99d8c9;
              color: white;
              vertical-align: top;
            }
        </style>
        """
        
        
    def click_button_widget(
            self, 
            desc: str, 
            margin: str='10px 0px 0px 10px',
            width='320px'
            ):
        '''
        Create a widget to show the button to click
        
        Args:
            desc: description to display on the button widget
            margin: top, right, bottom and left margins for the button widget
        '''
        # widget to show the button to click
        button = widgets.Button(description=desc, 
                                layout=Layout(margin=margin, width=width),
                                style=dict(font_style='italic',
                                           font_weight='bold'))
        
        # the output after clicking the button
        out = widgets.Output()
        
        return button, out
    
    
    def check_file_size(self, file):
        '''
        Function to check the uploaded file size
        
        Args:
            file: the uploaded file containing the text data
        '''
        # check total uploaded file size
        total_file_size = sum([i['metadata']['size'] for i in self.file_uploader.value.values()])
        print('The total size of the upload is {:.2f} MB.'.format(total_file_size/1000000))
        
        # display warning for individual large files (>1MB)
        large_text = [text['metadata']['name'] for text in self.file_uploader.value.values() \
                      if text['metadata']['size']>self.large_file_size and \
                          text['metadata']['name'].endswith('.txt')]
        if len(large_text)>0:
            print('The following file(s) are larger than 1MB:', large_text)
        
        
    def extract_zip(self, zip_file):
        '''
        Load zip file
        
        Args:
            zip_file: the file containing the zipped data
        '''
        # create an input folder if not already exist
        os.makedirs('input', exist_ok=True)
        
        # read and decode the zip file
        temp = io.BytesIO(zip_file['content'])
        
        # open and extract the zip file
        with ZipFile(temp, 'r') as zip:
            # extract files
            print('Extracting {}...'.format(zip_file['metadata']['name']))
            zip.extractall('./input/')
        
        # clear up temp
        temp = None
    
    
    def load_txt(self, file) -> list:
        '''
        Load individual txt file content and return a dictionary object, 
        wrapped in a list so it can be merged with list of pervious file contents.
        
        Args:
            file: the file containing the text data
        '''
        try:
            # read the unzip text file
            with open(file) as f:
                temp = {'text_name': file.name[:-4],
                        'text': f.read()
                }
            os.remove(file)
        except:
            file = self.file_uploader.value[file]
            # read and decode uploaded text
            temp = {'text_name': file['metadata']['name'][:-4],
                    'text': codecs.decode(file['content'], encoding='utf-8', errors='replace')
            }
            
            # check for unknown characters and display warning if any
            unknown_count = temp['text'].count('�')
            if unknown_count>0:
                print('We identified {} unknown character(s) in the following text: {}'.format(unknown_count, file['metadata']['name'][:-4]))
        
        return [temp]


    def load_table(self, file) -> list:
        '''
        Load csv or xlsx file
        
        Args:
            file: the file containing the excel or csv data
        '''
        if type(file)==str:
            file = self.file_uploader.value[file]['content']

        # read the file based on the file format
        try:
            temp_df = pd.read_csv(file)
        except:
            temp_df = pd.read_excel(file)
        
        # remove file from directory
        if type(file)!=bytes:
            os.remove(file)
            
        # check if the column text and text_name present in the table, if not, skip the current spreadsheet
        if ('text' not in temp_df.columns) or ('text_name' not in temp_df.columns):
            print('File {} does not contain the required header "text" and "text_name"'.format(file['metadata']['name']))
            return []
        
        # return a list of dict objects
        temp = temp_df[['text_name', 'text']].to_dict(orient='index').values()
        
        return temp
    
    
    def hash_gen(self, temp_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create column text_id by md5 hash of the text in text_df
        
        Args:
            temp_df: the temporary pandas dataframe containing the text data
        '''
        #temp_df['text_id'] = temp_df['text'].apply(lambda t: hashlib.md5(t.encode('utf-8')).hexdigest())
        temp_df['text_id'] = temp_df['text'].apply(
            lambda t: hashlib.shake_256(t.encode('utf-8')).hexdigest(5))
        
        return temp_df
    
    
    def process_upload(self, deduplication: bool = True):
        '''
        Pre-process uploaded .txt files into pandas dataframe

        Args:
            deduplication: option to deduplicate text_df by text_id
        '''
        # create placeholders to store all texts and zipped file names
        all_data = []; zip_files = []
        
        # read and store the uploaded files
        files = list(self.file_uploader.value.keys())
        
        # extract zip files (if any)
        for file in files:
            if file.lower().endswith('zip'):
                self.extract_zip(self.file_uploader.value[file])
                zip_files.append(file)
        
        # remove zip files from the list
        files = list(set(files)-set(zip_files))
        
        # add extracted files to files
        for file_type in ['*.txt', '*.xlsx', '*.csv']:
            files += [file for file in Path('./input').rglob(file_type) if 'MACOSX' not in str(file)]
        
        print('Reading uploaded files...')
        print('This may take a while...')
        # process and upload files
        for file in tqdm(files):
            # process text files
            if str(file).lower().endswith('txt'):
                text_dic = self.load_txt(file)
                    
            # process xlsx or csv files
            else:
                text_dic = self.load_table(file)
            all_data.extend(text_dic)
        
        # remove files and directory once finished
        os.system('rm -r ./input')
        
        # convert them into a pandas dataframe format and add unique id
        self.text_df = pd.DataFrame.from_dict(all_data)
        self.text_df = self.hash_gen(self.text_df)
        
        # clear up all_data
        all_data = []; zip_files = []
        
        # deduplicate the text_df by text_id
        if deduplication:
            self.text_df.drop_duplicates(subset='text_id', keep='first', inplace=True)
    
    
    def calculate_similarity(self, 
                             ngram_value: int = 1, 
                             num_perm: int = 256, 
                             similarity_cutoff: float = 0.5, 
                             actual_jaccard: bool = False):
        '''
        Function to calculate/estimate Jaccard similarity between documents in the corpus
        and begin the process of deduplicating based on the specified parameters

        Args:
            ngram_value: the n-gram size (the number of words used to detect similarity)
            num_perm: the number of permutation functions for estimating Jaccard similarity
            similarity_cutoff: the Jaccard similarity cut-off for determining similar documents
            actual_jaccard: whether to calculate actual or estimated Jaccard similarity
        '''
        try:
            tqdm.pandas()
            
            # Step 1: calculate word counts
            print('Step 1/9...')
            self.text_df['word_count'] = self.text_df.progress_apply(
                lambda x: self.count_text_words(x.text), 
                axis=1)
            
            # Step 2: create text hash (to be used for estimating Jaccard similarity)
            print('Step 2/9...')
            self.text_df['hash'] = self.text_df.progress_apply(
                lambda x: self.make_text_hash(x.text, 
                                              num_perm, 
                                              ngram_value), 
                axis=1)
            
            # Step 3 and 4: identify similar documents based on estimated Jaccard similarity
            print('Step 3/9...')
            # Create LSH index
            lsh = MinHashLSH(threshold=similarity_cutoff, num_perm=num_perm)
            
            for index, row in tqdm(self.text_df.iterrows(), 
                                   total=len(self.text_df)):
                lsh.insert(row['text_id'], row['hash'])
            
            print('Step 4/9...')
            self.text_df['matched_list'] = self.text_df.progress_apply(
                lambda x: self.get_matches(lsh, 
                                           x.hash, 
                                           x.text_id), 
                axis=1)
            
            # Step 5: calculate actual or estimate Jaccard similarity
            print('Step 5/9...')
            self.text_df['jaccards'] = self.text_df.progress_apply(
                lambda x: self.get_jaccards(
                    df=self.text_df, 
                    original=x.text_id, 
                    matched_list= x.matched_list, 
                    ngram_value=ngram_value,
                    actual_jaccard=actual_jaccard), axis=1)
            
            # Step 6 & 7: collating list of similar documents
            print('Step 6/9...')
            intermediate_df = self.text_df[['matched_list', 
                                            'text_id', 
                                            'text_name',
                                            'jaccards']].copy()
            intermediate_df['listlen'] = intermediate_df.progress_apply(
                lambda x: len(x.matched_list), axis = 1)
            
            print('Step 7/9...')
            intermediate_df = intermediate_df[intermediate_df.listlen > 0]
            intermediate_df['text_id_duped'] = intermediate_df.progress_apply(
                lambda x: [x.text_id] * x.listlen, axis = 1)
            
            self.deduplication_df = pd.DataFrame(
                {
                    'text_id2' : self.explode_list(intermediate_df, 'matched_list'),
                    'similarity': self.explode_list(intermediate_df, 'jaccards'),
                    'text_id1': self.explode_list(intermediate_df, 'text_id_duped'),
                }
            )
            
            # join with article dates (from metadata)
            metadata_df = self.text_df[['text_id', 'text_name', 'word_count']].copy()
            
            self.deduplication_df = self.deduplication_df.merge(
                metadata_df,
                left_on='text_id1', 
                right_on='text_id', 
                how='inner').drop('text_id', axis=1).merge(
                metadata_df, 
                left_on='text_id2', 
                right_on='text_id',
                suffixes=('1', '2'),
                how='left').drop('text_id', axis=1)
            
            # Step 8: removing duplication from list of similar documents
            print('Step 8/9...')
            keep_index = {'index': [],
                          'text_pair': []}
            for index, row in tqdm(self.deduplication_df.iterrows(), 
                                   total=len(self.deduplication_df)):
                if set([row.text_id1,row.text_id2]) not in keep_index['text_pair']:
                    keep_index['index'].append(index)
                    keep_index['text_pair'].append(set([row.text_id1,row.text_id2]))
            
            self.deduplication_df = self.deduplication_df[self.deduplication_df.index.isin(keep_index['index'])]
            
            self.similar_doc_id = self.get_duplicate_ids(
                self.deduplication_df,
                similarity_cutoff)
            
            # Step 9: recommendation to keep or remove and deduplciate documents
            print('Step 9/9...')
            status1 = []; status2 = []
            for index, row in tqdm(self.deduplication_df.iterrows(), 
                                   total=len(self.deduplication_df)):
                if row.text_id1 in self.similar_doc_id:
                    status1.append('remove')
                else:
                    status1.append('keep')
                if row.text_id2 in self.similar_doc_id:
                    status2.append('remove')
                else:
                    status2.append('keep')
            
            self.deduplication_df['status1'] = status1
            self.deduplication_df['status2'] = status2
            
            column_names = ['text_id1', 'text_name1', 'word_count1', 'status1',
                            'similarity',
                            'text_id2', 'text_name2', 'word_count2', 'status2']
            self.deduplication_df = self.deduplication_df.reindex(columns=column_names)
            self.deduplication_df = self.deduplication_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
            
            self.deduplicated_text_df = self.text_df[~self.text_df.text_id.isin(self.similar_doc_id)]
        
        except:
            print('No similar documents found. You can try using lower simiarity cutoff to find similar documents...')
        
    
    def display_deduplication_list(self): 
        '''
        Function to display deduplication text list 
        '''
        # display in html format for styling purpose
        df_html = self.deduplication_df.to_html(escape=False)
        
        # Concatenating to single string
        df_html = self.style+'<div class="dataframe-div">'+df_html+"\n</div>"
        
        # display the pair of similar texts
        display(HTML(df_html))
        
        
    def update_list(self, 
                    index: int, 
                    item: list):
        '''
        Function to update duplicated documents list based on selected action
        (whether to 'keep' or 'remove' duplicated documents)

        Args:
            index: the row index of the pair of documents being reviewed
            item: a list containing text_id and status ('keep' or 'remove') of the documents
        '''
        if self.deduplication_df[item[1]][index]=='remove':
            if self.deduplication_df[item[0]][index] not in self.similar_doc_id:
                self.similar_doc_id.append(self.deduplication_df[item[0]][index])
        else:
            if self.deduplication_df[item[0]][index] in self.similar_doc_id:
                self.similar_doc_id.remove(self.deduplication_df[item[0]][index])
        
        
    def display_deduplication_text(self):
        '''
        Function to display pair of possible duplicated texts
        '''
        # output the list of identified duplicated texts and the recommendations
        list_out = widgets.Output()
        
        with list_out:
            self.display_deduplication_list()
            
        # widgets for selecting the row index containing pair of similar documents to review
        enter_index, index = self.select_n_widget('<b>Select row index:</b>', 0)
        
        # widget to display pair of similar documents to review
        display_button, display_out = self.click_button_widget(desc='Display pair of texts', 
                                                       margin='20px 0px 10px 0px',
                                                       width='150px')
        
        # options on what to do with pair of similar documents
        act_options = ['keep left text only', 'keep right text only', 
                       'keep both', 'remove both']
        
        # default action shown based on recommendation by the tool
        default_action = {'keep & remove':act_options[0],
                          'remove & keep':act_options[1],
                          'keep & keep':act_options[2],
                          'remove & remove':act_options[3]}
        
        # widget to select action
        enter_action, select_action = self.select_options('<b>Select action:</b>',
                                                          ['None'],
                                                          'None')
        
        # widget to update selection based on selected action
        update_button, update_out = self.click_button_widget(desc='Update selection', 
                                                       margin='20px 0px 10px 0px',
                                                       width='150px')
        
        # function to define what happens when the display button is clicked
        def on_display_button_clicked(_):
            with display_out:
                clear_output()
                text_pair = self.deduplication_df[
                    self.deduplication_df.index == index.value].iloc[0,:].squeeze()
                self.show_comparison(text_pair)
                
                select_action.options = act_options
                select_action.value = default_action['{} & {}'.format(text_pair.status1,
                                                                      text_pair.status2)]
        
        # link the display_button with the function
        display_button.on_click(on_display_button_clicked)
        
        # function to define what happens when the update button is clicked
        def on_update_button_clicked(_):
            with update_out:
                clear_output()
                chosen_action = select_action.value
                for k, v in default_action.items():
                    if chosen_action==v:
                        self.deduplication_df.iloc[index.value,3]=k.split(' & ')[0]
                        item = ['text_id1','status1']
                        self.update_list(index.value, item)
                        self.deduplication_df.iloc[index.value,-1]=k.split(' & ')[1]
                        item = ['text_id2','status2']
                        self.update_list(index.value, item)
                        
            with list_out:
                clear_output()
                self.display_deduplication_list()
                
            with display_out:
                clear_output()
                text_pair = self.deduplication_df[
                    self.deduplication_df.index == index.value].iloc[0,:].squeeze()
                self.show_comparison(text_pair)
                    
        # link the update_button with the function
        update_button.on_click(on_update_button_clicked)
        
        # displaying inputs, buttons and their outputs
        vbox1 = widgets.VBox([enter_index, 
                              index, 
                              display_button],
                             layout = widgets.Layout(width='280px', height='130px'))
        
        vbox2 = widgets.VBox([enter_action, 
                              select_action, 
                              update_button],
                             layout = widgets.Layout(width='280px', height='130px'))
        hbox1 = widgets.HBox([vbox1, vbox2],
                             layout = widgets.Layout(width='800px'))
        
        vbox3 = widgets.HBox([display_out],
                             layout = widgets.Layout(height='300px'))
        
        vbox = widgets.VBox([list_out, hbox1, update_out, vbox3])
        
        return vbox
    
    
    def finalise_and_save(self, n: int):
        '''
        Function to finalise deduplication selections and save all kept texts 
        
        Args:
            n: the number of rows to display
        '''
        # output the list of non-duplicated texts
        deduplicated_out = widgets.Output()
        
        with deduplicated_out:
            self.deduplicated_text_df = self.text_df[~self.text_df.text_id.isin(self.similar_doc_id)]
            display(self.deduplicated_text_df.iloc[:,0:4].head(n))
            
        # widget to save non-duplicated texts
        save_button, save_out = self.click_button_widget(desc='Save texts', 
                                                       margin='20px 0px 10px 0px',
                                                       width='150px')
            
        # function to define what happens when the save button is clicked
        def on_save_button_clicked(_):
            with save_out:
                clear_output()
                # create an output folder if not already exist
                os.makedirs('./output/saved_files', exist_ok=True)
                
                for index, row in tqdm(self.deduplicated_text_df.iterrows(), 
                                       total=len(self.deduplication_df)):
                    with open('./output/saved_files/{}_{}.txt'.format(row.text_id, 
                                                                      row.text_name), 'w') as f:
                        f.write(row.text)
                    
                def zipdir(path, ziph):
                    # ziph is zipfile handle
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            ziph.write(os.path.join(root, file), 
                                       os.path.relpath(os.path.join(root, file), 
                                                       os.path.join(path, '..')))

                with zipfile.ZipFile('./output/deduplicated_texts.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipdir('./output/saved_files/', zipf)
                
                # remove files and directory once finished
                os.system('rm -r ./output/saved_files')
                print('Deduplicated texts saved. Click below to download:')
                
                # download the zip file onto your computer
                file_name = './output/deduplicated_texts.zip'
                display(DownloadFileLink(file_name, file_name[9:]))
        
        # link the save_button with the function
        save_button.on_click(on_save_button_clicked)
        
        # displaying inputs, buttons and their outputs
        vbox = widgets.VBox([deduplicated_out, save_button, save_out])
        
        return vbox
    
    
    def show_comparison(self, text_pair: pd.Series):
        '''
        Function to display comparison of pair of similar texts side-by-side in html format
        
        Args:
            text_pair: the pair of texts to display
        '''
        # obtain text and metadata
        title1 = f'Text: {text_pair.text_name1}'
        title2 = f'Text: {text_pair.text_name2}'
        text1 = self.text_df[self.text_df['text_id']==text_pair.text_id1].text.to_list()[0]
        text2 = self.text_df[self.text_df['text_id']==text_pair.text_id2].text.to_list()[0]
        
        metadata1 = f'text_id: {text_pair.text_id1}; word_count: {text_pair.word_count1}; Jaccard similarity: {text_pair.similarity}; status: {text_pair.status1}'
        metadata2 = f'text_id: {text_pair.text_id2}; word_count: {text_pair.word_count2}; Jaccard similarity: {text_pair.similarity}; status: {text_pair.status2}'
        
        myhtml = html_diffs(text1, text2, title1, title2, metadata1, metadata2)
        
        display(HTML(myhtml))
        
        
    def count_text_words(self, text: str):
        '''
        Function to tokenize a document and count the number of words in the document
        
        Args:
            text: the text to tokenize and count the number of words
        '''
        return len(list(tokenize(text)))
    
    
    def convertTuple(self, tup: tuple):
        '''
        Function to join a tuple of words into a sentence
        
        Args:
            tup: the tuple containing the list of words
        '''
        return ' '.join(tup)
    
    
    def make_text_hash(self, 
                       text: str, 
                       num_perm: int = 256, 
                       ngram_value: int = 1):
        '''
        Function to create hash for each document using datasketch.MinHash
        (https://ekzhu.com/datasketch/minhash.html)
        
        Args:
            text: the text to create hash
            num_perm: the number of permutation functions for estimating Jaccard similarity
            ngram_value: the n-gram size (the number of words used to detect similarity)
        '''
        # tokenize text, obtain ngrams, convert to a list, 
        # join the items in the list and get the tuple of the new list
        myset = set([self.convertTuple(x) for x in list(ngrams(tokenize(text), ngram_value))])
        
        # initiate MinHash and set the number of permutation functions used in MinHash
        hash1 = MinHash(num_perm)
        
        # get minhash object from the set
        for d in myset:
            hash1.update(d.encode('utf8'))
            
        return hash1
    
    
    def get_matches(self, 
                    lsh, 
                    hash_doc, 
                    text_id: str) -> list:
        '''
        Function to find matched documents
        
        Args:
            lsh: the Locality Sensitive Hashing (LSH) index 
            hash_doc: the hash for the document
            text_id: the text id of the document
        '''
        # approximate neighbours with Jaccard similarity > the set MinHashLSH threshold (in this case 0.5)
        matches = lsh.query(hash_doc)
        
        # remove if article ID is the same (the same document)
        matches.remove(text_id)
        
        return matches
    
    
    def find_jaccard(self, 
                     set1: set, 
                     set2: set, 
                     m1, 
                     m2, 
                     actual_jaccard: bool = False):
        '''
        Function to find matched documents
        
        Args:
            set1: set of words from the first document
            set2: set of words from the second document
            m1: MinHash from the first document
            m2: MinHash from the second document
            actual_jaccard: whether to calculate actual or estimated Jaccard similarity
        '''
        # calculate jaccard similarity
        if len(set1.union(set2)) != 0:
            if actual_jaccard:
                return len(set1.intersection(set2))/len(set1.union(set2))
            else:
                return m1.jaccard(m2)
        else:
            # the sets have nothing in common
            # to avoid divide by 0 error
            return 0
    
    
    def get_jaccards(self, 
                     df: pd.DataFrame, 
                     original: str, 
                     matched_list: list, 
                     ngram_value: int, 
                     actual_jaccard: bool = False):
        '''
        Function to find matched documents
        
        Args:
            df: the pandas DataFrame containing the texts
            original: the text id of the first document
            matched_list: a list of text id's of the matched (possible similar) documents
            ngram_value: the n-gram size (the number of words used to detect similarity)
            actual_jaccard: whether to calculate actual or estimated Jaccard similarity
        '''
        # get ngrams and set for the seletced article_id
        body1 = df[(df['text_id'] == original)]['text'].values[0].lower()
        set1 = set(nltk.ngrams(tokenize(body1), n=ngram_value))
        jaccards = []
        m1 = df[(df['text_id'] == original)]['hash'].values[0]
        
        # no matches for this article
        if len(matched_list) == 0:
            return []
        
        else:
            # if matches, calculate the jaccard similarity between the sets
            for id in matched_list:
               body2 = df[(df['text_id'] == id)]['text'].values[0].lower()
               set2 = set(nltk.ngrams(tokenize(body2), n=ngram_value))
               m2 = df[(df['text_id'] == id)]['hash'].values[0]
               jaccard = round(self.find_jaccard(set1, set2, m1, m2, actual_jaccard),4)
               jaccards.append(jaccard)
            
            return jaccards
    
    
    def explode_list(self, 
                     df: pd.DataFrame, 
                     col: 'str') -> list:
        '''
        Function to convert a list of list to a flat list
        
        Args:
            df: the pandas DataFrame containing the texts
            col: the column in the pandas DataFrame to be converted into a flat list
        '''
        return list(chain.from_iterable(df[col].to_list()))
    
    
    def plot_hash_similarity_by_source(self, df: pd.DataFrame):
        '''
        Function to plot a histogram of similarity count
        
        Args:
            df: the pandas DataFrame containing the similarity
        '''
        # %% Visualise similarity scores
        title = "Similarity count accross the entire corpus"
    
        plot = sns.histplot(data=(df[
            # return single row for article_id and similarity_score,
            # so one row per article for this plot    
            ~df[
            ['text_id1',"similarity"]]
            .duplicated()]) , x="similarity").set_title(title)
        
        return plot
    
    
    def get_duplicate_ids(self, 
                          df: pd.DataFrame, 
                          min_similarity: float) -> list:
        '''
        Function to obtain duplicated text id's based on similarity cutoff oand word count
        
        Args:
            df: the pandas DataFrame containing the texts
            min_similarity: Jaccard similarity cutoff for determining similar documents
        '''
        df = df[df.similarity >= min_similarity]
        df = df[df.word_count1 >= df.word_count2]
        
        list1 = list(df['text_id1'].values)
        list2 = list(df['text_id2'].values)
        assert len(list1) == len(list2)
        
        considered, drop = set(), []
        
        for i in range(len(list1)):
            if list1[i] not in considered:
                considered.add(list1[i])
                considered.add(list2[i])
                drop.append(list2[i])
            else:
                if list2[i] not in considered:
                    considered.add(list2[i])
                    drop.append(list2[i])
        drop = sorted(list(set(drop)))
        
        return drop
    
    
    def select_n_widget(self, 
                        instruction: str, 
                        value: int):
        '''
        Create widgets for selecting a number
        
        Args:
            instruction: text instruction for user
            value: initial value of the widget
        '''
        # widget to display instruction
        enter_n = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widgets for selecting n
        n_option = widgets.BoundedIntText(
            value=value,
            min=0,
            #max=len(self.text_df),
            step=1,
            description='',
            disabled=False,
            layout = widgets.Layout(width='150px')
        )
        
        return enter_n, n_option
    
    
    def select_options(self, 
                       instruction: str,
                       options: list,
                       value: str):
        '''
        Create widgets for selecting options
        
        Args:
            instruction: text instruction for user
            options: list of options for user
            value: initial value of the widget
        '''
        # widget to display instruction
        enter_text = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widget to select entity options
        select_option = widgets.Dropdown(
            options=options,
            value=value,
            description='',
            disabled=False,
            layout = widgets.Layout(width='150px')
            )
        
        return enter_text, select_option
    
    
    def click_button_widget(
            self, 
            desc: str, 
            margin: str='10px 0px 0px 10px',
            width='320px'
            ):
        '''
        Create a widget to show a button to click
        
        Args:
            desc: description to display on the button widget
            margin: top, right, bottom and left margins for the button widget
            width: the width of the button
        '''
        # widget to show the button to click
        button = widgets.Button(description=desc, 
                                layout=Layout(margin=margin, width=width),
                                style=dict(font_weight='bold'))
        
        # the output after clicking the button
        out = widgets.Output()
        
        return button, out