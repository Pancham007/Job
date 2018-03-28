#b. Naïve Bayesian Classification (Bigram) based POS Tagging:
#c. Apply model (a) and (b) on the sentence below, and show the difference in error rates.

from collections import defaultdict
from prettytable import PrettyTable

file = 'HW2_S18_NLP6320_POSTaggedTrainingSet-Windows.txt'
input_sentence = "The president wants to control the board 's control"

bigrams = defaultdict(int)
tag_c = defaultdict(int)
tag_c['null'] = 1
word_tag = defaultdict(int)
#**** Tag List ****
tag_list = ["PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB","$","#","\"","(",")",",",".",":"]

with open(file,"r") as f:
   input_file=f.read()

tokens = input_file.split(" ")

for i in range(0, len(tokens)-1):
    if '_' in tokens[i] and '_' in tokens[i+1]:
        curr_word = tokens[i].split("_")[0]
        tag_current = tokens[i].split("_")[1]
        next_word = tokens[i+1].split("_")[0]
        tag_next = tokens[i+1].split("_")[1]

        bigrams[tag_current+","+tag_next] += 1
        tag_c[tag_current] += 1
        word_tag[curr_word+","+tag_current] += 1

if '_' in tokens[i+1]:
    tag_c[tokens[i+1].split("_")[1]] += 1
    word_tag[tokens[i+1].split("_")[0]+","+tokens[i+1].split("_")[1]] += 1

#********* Evaluating Probabilities *********

sentence = input_sentence
s_tokens = sentence.split(" ")
distinct_tokens = []
distinct_val = []
distinct_values_for_probabilities = []

nb_error_rate = 0
#****** Generating table *****
tab = PrettyTable()
tab.field_names = ["word", "correct_tag","unigram_probable_tag","final_prob","p(word|tag)","p(curr_tag|prev_tag)","p(next_tag|curr_tag)","expected_next_tag"]
#***** comparing previous tag with current tag ********
count = 0
tag_previous = "NN"
for tags in tag_list:
    tc = word_tag[s_tokens[0]+","+tags]
    if tc > count:
        count = tc
        tag_previous = tags

# print("tag_previous: "+str(tag_previous))
tab.add_row([s_tokens[0], tag_previous, tag_previous, 1, 1, 1, 1, tag_previous])

for i in range(1, len(s_tokens)):
    curr_word = s_tokens[i]
    final_prob = 0.0
    final_p_word_tag = 0.0
    final_pr_prev_curr = 0.0
    final_pr_curr_next = 0.0
    new_curr_tag = "NN"
    new_next_tag = "NN"
    for curr_tag in tag_list:
        for next_tag in tag_list:
            p_word_tag = word_tag[curr_word+","+curr_tag]/(tag_c[curr_tag]+0.0001)
            pr_prev_curr = bigrams[tag_previous+","+curr_tag]/(tag_c[tag_previous]+0.0001)

            if i == len(s_tokens)-1:
                pr_curr_next = 1
            else:
                pr_curr_next = bigrams[curr_tag+","+next_tag]/(tag_c[curr_tag]+0.0001)

            prob = p_word_tag*pr_prev_curr*pr_curr_next
            if prob>final_prob:
                final_p_word_tag = p_word_tag
                final_pr_prev_curr = pr_prev_curr
                final_pr_curr_tag = pr_curr_next
                final_prob = prob
                new_curr_tag = curr_tag
                new_next_tag = next_tag

    tag_t = "--"
    count = 0
    for tags in tag_list:
        tc = word_tag[curr_word+","+tags]
        if tc > count:
            count = tc
            tag_t = tags

    if tag_t != new_curr_tag:
        # print("tag_t: "+str(tag_t)+" curr_tag: "+str(new_curr_tag))
        nb_error_rate = nb_error_rate + 1

    tab.add_row([s_tokens[i], new_curr_tag, tag_t, final_prob, final_p_word_tag, final_pr_prev_curr, final_pr_curr_next, new_next_tag])

    #set current tag as tag previous
    tag_previous = new_curr_tag

print(tab)
print("Naive Bayesian Error rate: "+str(float(nb_error_rate/len(tokens)))+"\n")
