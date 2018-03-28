'''Generating a POS tagged set for all the words in the corpus
Output_POS_tag.txt- dictionary of all the words tagged along with the count'''

from collections import defaultdict
import re

file = 'HW2_S18_NLP6320_POSTaggedTrainingSet-Windows.txt'

Unigrams = []
UnigramsWithCount = defaultdict(int)

words = re.findall('\S+', open(file).read())
for i in range(0, len(words) - 1):
    Unigrams = words[i].split("_")
    # print TaggedUnigrams
    for j in range(0, len(Unigrams) - 1):
        UnigramsWithCount[Unigrams[j], Unigrams[j + 1]] += 1

total_count = len(words)
print (UnigramsWithCount)

output_file = open('Output_POS_tag.txt', 'w')
for (word, occurance) in UnigramsWithCount.items():
    output_file.write(('{:25}{:10}'.format(str(word), occurance))+"\n");

output_file.close();