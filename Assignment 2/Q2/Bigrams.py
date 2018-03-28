"""
Compute the sentence probability under the three following scenarios:
i. Use the bigram model without smoothing.
ii. Use the bigram model with add-one smoothing
iii. Use the bigram model with Good-Turing discounting
"""
from collections import defaultdict

Corpus = 'HW2_S18_NLP6320-NLPCorpusTreebank2Parts-CorpusA-Windows.txt'

Bigrams = defaultdict(int)
unigrams = {}
counts = []
#Unigrams
with open(Corpus, 'r') as C:
    for sentence in C:
        word = sentence.split()
        for x in range(0, len(word) - 1):
            Bigrams[word[x], word[x + 1]] += 1

#print(Bigrams)
#Bigrams
with open(Corpus, 'r') as C:
    for sentence in C:
        word = sentence.split()
        for x in word:
            if x not in unigrams:
                unigrams[x] = 1
            else:
                unigrams[x] = unigrams[x] + 1
#print(unigrams)

#Total Count of Bigrams
Total_Bigram_Count = len(Bigrams);
#print ("Total No of Bigrams: "+ str(Total_Bigram_Count))

#Writing values in file
output_file = open('Output_Bigrams.txt', 'w')
output_file.write(('{:59}{:35}{:45}{:45}'.format('Bigram', 'Count', 'P(Without Smoothing)', 'P(Add-one Smoothing)'))+"\n")
output_file.write('*' * 160+"\n")
for (x, count) in Bigrams.items():
    unigram_Count = unigrams.get(x[1])
    bigram_Count = Bigrams.get(x)

    row = ('{:30}{:35}{:45}{:45}'.format(str(x), count, (count / unigram_Count),((count + 1) / (unigram_Count + Total_Bigram_Count)),))

    print (row)
    output_file.write(('{:30}{:35}{:45}{:45}'.format(str(x),count,(count/unigram_Count),((count+1)/(unigram_Count+Total_Bigram_Count)))) + "\n")

output_file.close();

#***** Probabilities ******

sentence = input("Enter Sentence:\n")
Bigrams = defaultdict(int)
unigrams = {}
words = sentence.split()
for i in range(0, len(words) - 1):
    Bigrams[words[i], words[i + 1]] += 1

for word in words:
    if word not in unigrams:
        unigrams[word] = 1
    else:
        unigrams[word] = unigrams[word] + 1

print('{:29}{:25}{:30}{:30}'.format('Bigram', 'Count', 'P(Without Smoothing)', 'P(Add-one Smoothing)'))
print('*' * 100)
total_count = len(Bigrams);

for (word, counts) in Bigrams.items():
    Unigram_Count = unigrams.get(word[1])
    Bigram_Count = Bigrams.get(word)
    # Printing Bigrams and occurences and Probabilty(Count(A^B)/Count(B)) and Add 1(Count(A^B)+1/Count(B)+V)
    out = (('{:15}{:20}{:30}{:30}'.format(str(word), counts, (counts / Unigram_Count),((counts + 1) / (Unigram_Count + total_count)))));
    print (out);