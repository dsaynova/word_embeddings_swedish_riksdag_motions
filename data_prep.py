#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict
from bs4 import BeautifulSoup
from os import listdir
import codecs
import string
import re
import csv
import pickle
import argparse



def extract_party(p):
    start = p.find("(") + len("(")
    end = p.find(")")
    #EXTRACT TEXT BETWEEN PARENTHESIS
    substring = p[start:end].strip('båda ')
    party = tuple(substring.lower().split(","))
    return party


def extract_data(fn):
    
    f=codecs.open(fn, 'r', encoding='utf-8-sig')
    document= BeautifulSoup(f.read(), "lxml")

    text = ''
    
    #EXTRACT INFO FROM ALL PARAGRAPH TAGS UNLESS NAMES OF SIGNEES
    paragraphs = document.find_all('p')
    for p in paragraphs:
        if p.has_attr('class') and p['class'][0] != 'Underskrifter':
            text = text + ' ' + p.get_text().replace('\xad', '')#remove soft-hyphens
        elif not p.has_attr('class'):
            text = text + ' ' + p.get_text().replace('\xad', '')
    
    #TRY TO EXTREACT DATE INFO
    try:   
        ds = document.find('span','sidhuvud_beteckning')
        if ds == None: ds = document.find('div','sidhuvud_beteckning')
        date = ds.string
        date = date[0:7]
    except: date = None
    
    #TRY TO EXTREACT PARTY INFO
    try:
        ws = document.find('span','MotionarLista')
        if ws == None: ws = document.find('div','MotionarLista')
        written_by = ws.string
        #EXTRACT PARTY ONLY
        written_by = extract_party(written_by)
    except: written_by = None

    #REMOVE EMPTY LINES
    text_remove_empty = ''
    for p in text.split('\n'):
        if p.strip() != '':
            text_remove_empty = text_remove_empty + ' ' + p
    
    return date, written_by, text_remove_empty



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=False)
    args = parser.parse_args()


    #LIST OF ALL CSV FILES
    filenames = listdir(args.data_folder)
    csv_files = [filename for filename in filenames if filename.endswith(".csv")]

    #GET METADATA FROM CSV FILES
    meta_data = defaultdict()
    for name in csv_files:
        with open(args.data_folder+name, 'r', encoding='utf-8-sig') as f:
            metareader = csv.reader(f, quotechar='"', delimiter=',')
            
            for line in metareader:
                #EXTRACT FILE NAME, DATE, WRITTEN BY LINE
                n, session, d, wb = line[1], line[2][0:4], line[11], line[14]
                
                if n == 'GZ02K324': 
                    wb = line[15] #data issue
                    
                #EXTRACT PARTY ONLY
                party = extract_party(wb)

                if party == ('',):
                    party = re.sub(r'[^a-zA-Z]', '', line[7]).lower()
                
                #ADD FLAG FOR REMOVED MOTIONS
                if line[13].strip().lower() == 'motionen utgår': 
                    deleted = True
                else: deleted = False
                

                meta_data[name.replace('.csv','')+'/'+n.replace("ö", "î").rstrip("-").lower()] = [d, party, deleted, int(session)]

    #READ MOTION TEXT
    data_1988_2009 = defaultdict()
    data_2010_2020 = defaultdict()

    for k, v in meta_data.items():

        #SELECT S AND M ONLY
        if all([j.lower().strip() not in ['m','s'] for j in v[1]]): continue
        #SKIP DELETED
        if v[2]: continue

        #SPLIT BY SESSION AND CHECH IF WITHIN STUDIED PERIOD
        if 1988<=v[3]<=2009 and int(v[0][0:4])>=1988:
            #FROM TXT
            text=''
            with open(args.data_folder+k+'.txt', 'r', encoding='utf-8-sig') as f:
                for line in f: text += ' ' + line.strip('\n').replace('\xad', '').translate(str. \
                        maketrans('', '', string.punctuation)).lower()
            if any([j.lower().strip() in ['m'] for j in v[1]]):
                data_1988_2009[k] = [v[0], 'm', text]
            if any([j.lower().strip() in ['s'] for j in v[1]]):
                data_1988_2009[k] = [v[0], 's', text]

        elif 2010<=v[3]:
            #FROM HTML - newer data has consistent html format
            _, _, text = extract_data(args.data_folder+k+'.html')
            if any([j.lower().strip() in ['m'] for j in v[1]]):
                data_2010_2020[k] = [v[0], 'm', text]
            if any([j.lower().strip() in ['s'] for j in v[1]]):
                data_2010_2020[k] = [v[0], 's', text]            



    #SAVE DATA FILE
    if args.output_folder:
        out = args.output_folder
    else: out = ''
    with open(out+"data_1988_2009.pkl","wb") as f:
        pickle.dump(data_1988_2009,f)
    with open(out+"data_2010_2020.pkl","wb") as f:
        pickle.dump(data_2010_2020,f)

if __name__ == "__main__":
    main()

