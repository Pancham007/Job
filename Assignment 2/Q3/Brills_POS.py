'''a. Transformation-based POS Tagging: Implement Brill’s transformation-based POS
tagging algorithm using ONLY the previous word’s tag to create transformation
rules. '''
from collections import Counter

file = 'HW2_S18_NLP6320_POSTaggedTrainingSet-Windows.txt'
with open(file,"r") as f:
   train_set=f.read()

print("****************** Brills Tagger is learning rules ******************")

words = []
tags = []

for sentence in train_set.split('\n'):
	for word in sentence.split():
		words.append(word.split('_')[0])
		tags.append(word.split('_')[1])

#print (words)
unigrams = {}
n_Tags = set(tags)

for i in range(len(words)):
	if not words[i] in unigrams:
		unigrams[words[i]] = [tags[i]]
	else:
		unigrams[words[i]].append(tags[i])

#********* Evaluating the most probable tag **********

def get_best_instance(dictionary):
	for key, value in dictionary.items():
		counter = Counter(value)
		max_val = counter.most_common()[0]
		dictionary[key] = max_val[0]
	return dictionary
get_best_instance = get_best_instance(unigrams)

#********** Most Probable Errors ************

def Probable_Errors(words, tags, dictionary):
	m_tags = []
	error = 0

	for word in words:
		m_tags.append(dictionary[word])
	for i in range(len(m_tags)):
		if m_tags[i] != tags[i]:
			error += 1
	return m_tags
m_tags = Probable_Errors(words, tags, get_best_instance)

#************** Applying Brills Algorithm and getting the transform **********

def brills_tagger(tags, mostProbableTags, n_Tags):
	template = {}
	m_tags = mostProbableTags[:]
	index = 0
  
	while index < 5:
		threshold = 0
		index+=1
		print ("Generating Probable Tag Set : " , index)
		for fromTag in n_Tags:
			for toTag in n_Tags:
				b_dictionary = {}

				if fromTag == toTag:
					continue
			#***** tag transform *****
				for pos in range(1,len(m_tags)):
					if tags[pos] == toTag and m_tags[pos] == fromTag:
						rule = (m_tags[pos-1], fromTag, toTag)
						if rule in b_dictionary:
							b_dictionary[rule] += 1
						else:
							b_dictionary[rule] = 1
					elif tags[pos] == fromTag and m_tags[pos] == fromTag:

						rule = (m_tags[pos-1], fromTag, toTag)
						if rule in b_dictionary:
							b_dictionary[rule] -= 1
						else:
							b_dictionary[rule] = -1

				if b_dictionary:
					max_Key = max(b_dictionary, key=b_dictionary.get)
					max_val = b_dictionary.get(max_Key)

					if max_val > threshold:
						threshold = max_val
						row = max_Key

		for i in range(len(m_tags)-1):
			if m_tags[i] == row[0] and m_tags[i+1] == row[1]:
				m_tags[i+1] = row[2]
		template[row] = threshold

	good_template = sorted(template.items(), key=lambda x: x[1], reverse=True)
	print ("Generated Template List : \n\n" , good_template)
	return good_template

final_templates = brills_tagger(tags, m_tags, n_Tags)

# ****** Running Test Case *******

input = "The_DT president_NN wants_VBZ to_TO control_VB the_DT board_NN 's_POS control_NN"

err = 0
words = []
tags = []
most_freq = []

for i in input.split():
    input_token,input_tag = i.split("_")
    words.append(input_token)
    tags.append(input_tag)

for i in range(len(words)):
	most_freq.append(get_best_instance[words[i]])

brill_in = most_freq[:]

for i in range(len(most_freq)-1):
	for key, val in final_templates:
		prev_tag = key[0]
		from_tag = key[1]
		to_tag = key[2]

		if brill_in[i] == prev_tag and brill_in[i+1] == from_tag:
			brill_in[i+1] = to_tag
			err +=1
			break

a =[i for i, item in enumerate(tags) if item in brill_in]
brills_error = len(tags) - len(a)
brills_error_rate = brills_error/100

print("Brills Error : " , brills_error)
print("Brills Error%  : " , brills_error_rate , "%")
