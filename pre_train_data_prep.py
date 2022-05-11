from os import listdir
import re
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=False)
    args = parser.parse_args()

    folder_names = [\
        'rskr-2000-2009', 'rskr-2014-2017', 'rskr-2018-2021',\
        'prop-1998-2001', 'prop-2002-2005', 'prop-2006-2009', 'prop-2010-2013', 'prop-2014-2017', 'prop-2018-2021',\
        'ip-1998-2001', 'ip-2002-2005', 'ip-2006-2009', 'ip-2010-2013', 'ip-2014-2017', 'ip-2018-2021']

    all_text = []
    for folder in folder_names:
        for file_name in listdir(args.data_folder+folder):
            file_text = ''
            with open(args.data_folder+folder+'/'+file_name, mode = 'r', encoding='UTF-8', errors = 'ignore') as f:
                for line in f.readlines():
                    if line.strip() and re.sub(r'[^\w\s]', '', re.sub(r'[0-9]', '', line)).strip():
                        file_text+=re.sub(r'[^\w\s]', '', line).strip().lower() + ' '

            all_text.append(file_text)

    #SAVE DATA FILE
    if args.output_folder:
        out = args.output_folder
    else: out = ''
    with open(out+'pre_train_text.txt', 'w') as f:
        for item in all_text:
            f.write("%s\n" % item)

if __name__ == "__main__":
    main()