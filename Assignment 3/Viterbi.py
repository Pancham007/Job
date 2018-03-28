"""
3. Programmatically implement the Viterbi algorithm and run it with the HMM in Figure 2
to compute the most likely weather sequence and probability for a given observation
sequence. Example observation sequences: 331, 122313, 331123312, etc.
"""

observations = input("Input Sequence: ");

#emission probabilities

em_prob = {
            'start': {'0': '0.8','1': '0.2'},
              '0': {'0': '0.7','1': '0.3'},
              '1': {'0': '0.4','1': '0.6'}
           }

#observation probabilities

obs_freq = {
    '0': {'1': '0.2','2': '0.4','3': '0.4'},
    '1': {'1': '0.5','2': '0.4','3': '0.1'}
            }

#states

hid_states = ['toh', 'dloc'];
obs_len = len(observations);
state_len = len(hid_states) + 2;

viterbi = [[0 for i in range(obs_len)] for y in range(state_len)]
pointer = [[0 for i in range(obs_len)] for y in range(state_len)]

for p, i in enumerate(hid_states):
    viterbi[p + 1][0] = float(em_prob['start'][str(p)]) * float(obs_freq[str(p)][str(observations[0])])
    pointer[p + 1][0] = 0

prob = [];

for i in range(1, len(observations)):
    for p_x, s in enumerate(hid_states):
        max_prob = 0
        index = 0
        for p_y, ss in enumerate(hid_states):
            #probability
            value = float(viterbi[p_y + 1][i - 1]) * float(em_prob[str(p_y)][str(p_x)]) * float(
                obs_freq[str(p_x)][observations[i]])
            if (value > max_prob):
                max_prob = value
                index = p_y + 1

        viterbi[p_x + 1][i] = max_prob;
        prob.append(max_prob);
        pointer[p_x + 1][i] = index

#Probability of most likely sequence

print("Probability: ")
print(prob[len(prob) - 2]);
weather_seq = ""
if (viterbi[1][len(observations) - 1] > viterbi[2][len(observations) - 1]):
    weather_seq = weather_seq + "toh ";
    trellis_state = 1;
    value = pointer[1][len(observations) - 1];
else:
    weather_seq = weather_seq + "dloc ";
    trellis_state = 2;
    value = pointer[1][len(observations) - 1];

len_obs = len(observations) - 1

#recovering the most likely sequence

while (len_obs > 0):
    if (value == 1):
        trellis_state = 1;
        weather_seq = weather_seq + "toh ";
    else:
        trellis_state = 2
        weather_seq = weather_seq + "dloc ";
    value = pointer[trellis_state][len_obs - 1];
    len_obs -= 1;

#Most likely weather sequence

print("Most Likely Weather Sequence: ", weather_seq[::-1])