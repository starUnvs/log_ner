import re
from glob import glob
from collections import defaultdict
import json
from tqdm import tqdm

MAX_LEN=1000000

DIG_SYMBOL ='χ' #'\u03c7'
S_SYMBOL='⸏'
#URL_SPLITER='\u00b6'
NAME_SPLITER='¯' #'\u00af'

def get_vocab(files):
	vocab=defaultdict(int)
	for f_path in files:
		with open(f_path,errors='ignore') as f:
			print('parsing '+f_path+'...')
			for line in tqdm(f.readlines()[:MAX_LEN]):
				words=parse(line)
				for word in words:
					vocab[word]+=1

		with open('./vocab_tmp.json','w') as f:
			json.dump(vocab,f,ensure_ascii=False)
	
	return vocab

def decode(tokens):
	return ''.join(tokens).replace('↑',' ')[1:]

def parse(text):
	# norm
	pt_func=re.compile(r'((?<=[a-z])(?=[A-Z]))|((?<=[A-Z])(?=[A-Z][a-z]))')
	pt_num2word=re.compile(r'(\W\d+)([a-zA-Z]+\W)')
	pt_word2num=re.compile(r'([\W|_][a-zA-Z]+)(\d+[\W|_])')
	pt_chinese=re.compile(r'[\u4e00-\u9fa5]')

	text=pt_func.sub('¯',text)
	text = pt_num2word.sub(r'\1¯\2',text)
	text = pt_word2num.sub(r'\1¯\2',text)
	text = pt_chinese.sub(r'Ĉ',text)
	#text=re.sub('\d',DIG_SYMBOL,text)
	#text=re.sub(r'://',URL_SPLITER,text)

	# tokenize
	pt_sep=re.compile(r'([Ĉ-_\{\}\\.: =,/()?~"%\n。，：+（\]）*&￥$#@!？|;；“”\'<》《>[】【]{1})|¯')

	words=re.split('\s+', text)
	words=['⸏'+word for word in words[1:]]
	for i in range(len(words)):
		words[i]=pt_sep.split(words[i])
		words[i]=[subword for subword in words[i] if subword!=None and subword!='']
	
	# postprocess
	pt_num1=re.compile(r'(^⸏?0x[0-9a-fA-F]+$)|((?=.*[0-9])(?=.*[a-fA-F])^⸏?[0-9a-fA-F]+$)')
	pt_num2=re.compile(r'^⸏?\d{5,}$')
	pt_digit=re.compile(r'\d')
	pt_all_digit=re.compile(r'^⸏?\d+$')

	tokens = [subword for subwords in words for subword in subwords]
	tokens = [pt_num1.sub('<NUM>',token) for token in tokens]
	tokens = [pt_num2.sub('<NUM>',token) for token in tokens]
	tokens = [pt_digit.sub('χ',token) if pt_all_digit.match(token)!=None else token for token in tokens]

#	tokens = [subword for subwords in words for subword in subwords]
#	tokens = [re.sub(r'(?=.*[0-9])(?=.*[a-f])[0-9a-f]+$','<NUM>',token) for token in tokens]
#	tokens = [re.sub(r'^↑?\d{5,}$','<NUM>',token) for token in tokens]
#	tokens = [re.sub(r'\d','χ',token) if re.match(r'^↑?\d+$',token)!=None else token for token in tokens]
	#tokens = [re.sub(r'^↑(0x)?[0-9a-fA-F]+$','↑<NUM>',token) for token in tokens]

	return tokens

if __name__ == '__main__':
	#files=glob('/data/logs_data/sys_data/*/*.log')
	files=glob('./sys_data/*/*.log')
	vocab=get_vocab(files)
	#vocab=get_vocab(glob('./datasets/kafka/kafka_10k.log'))
	vocab={k:v for k,v in sorted(vocab.items(),key=lambda x:x[1],reverse=True)}

	with open('./vocab.json','w') as f:
		json.dump(vocab,f,ensure_ascii=False)

	pass