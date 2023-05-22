import streamlit as st
import re
from model import sp_syllabler, onc_to_phon
import pickle
import time
from ussy import Ussy

piss_len = 45
shit_len = 19
piss_weights = 'piss_data/piss_weights.h5'
shit_weights = 'shit_data/shit_best_weights.h5'
piss_e2i = pickle.load(open('piss_data/e2i.pkl', 'rb'))
shit_e2i = pickle.load(open('shit_data/e2i.pkl', 'rb'))
shit_d2i = pickle.load(open('shit_data/d2i.pkl', 'rb'))

piss = sp_syllabler(piss_e2i, piss_len, 256, 256, len(piss_e2i) + 1)
shit = onc_to_phon(shit_e2i, shit_d2i, shit_len, 128, 500)

piss.model.load_weights(piss_weights)
shit.model.load_weights(shit_weights)

Ussy = Ussy(3, piss, shit)

st.set_page_config(page_title='Syllabify', page_icon=':pencil2:')

onsets = ['bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'sk', 'sl', 'sm', 'sn',
     'sp', 'st', 'str', 'sw', 'tr', 'ch', 'sh', 'm', 'c', 'b', 'r', 'd', 'h', 's', 'p', 'l', 'g', 'f', 'w', 't', 'k', 'n', 'v', 'st', 'pr', 'j', 'br', 'ch', 'gr', 'sh',
  'tr', 'cr', 'fr', 'z', 'sp', 'wh', 'cl', 'y', 'bl', 'th', 'fl', 'sch', 'pl', 'q', 'dr', 'str', 'sc', 'sl', 'kr', 'sw', 'gl',
  'ph', 'kl', 'sm', 'sn', 'kn', 'sk', 'mcc', 'scr', 'wr', 'mc', 'chr', 'spr', 'thr', 'tw', 'schw', 'mcg', 'mck', 'rh',
  'sq', 'schl', 'shr', 'schr', 'x', 'schm', 'mcm', 'gh', 'mcn', 'hyp', 'mccl', 'schn', 'mcd', 'hydr', 'kh', 'ts',
  'mcl', 'spl', 'dw', 'pf', 'mccr', 'mcf', 'typ', 'cz', 'sr', 'cycl', 'gn', 'hr', 'hy', 'syn', 'sz', 'kw', 'dyn', 'phys', 'symb', 'dyn', 'symb']

nuclei = ['a', 'e', 'i', 'o', 'u', 'oo', 'ia', 'ie', 'ee', 'io', 'au', 'ea', 'ou', 'ai', 'ue', 'ei', 'eau', 'eu', 'oe', 'ae', 'eo',
  'oa', 'oo', 'ao', 'ua', 'oi', 'ui', 'aa', 'ieu', 'uo', 'oia', 'aue', 'iu', 'aia', 'iou', 'ii', 'aio', 'uie', 'eia', 'iao' ,'y', 'uh', 
          'ay', 'ey','ah','eh','oh','oy', 'aigh', 'igh', 'eigh', 'aw', 'ow', 'ew', 'ye', 'ooh', 'owe', 'awe', 'ore', 'er', 'or', 'ere', 
          'are', 'ar', 'ur', 'ir', 'ire', 'ue', 'eye', 'aye', 'ye', 'uy']

coda = ['ct', 'ft', 'mp', 'nd', 'ng', 'nk', 'nt', 'pt', 'sk', 'sp', 'ss', 'st', 'ch', 'sh', 's', 'n', 'r', 'd', 'ng', 'l', 'rs', 'ns', 't', 'm', 'll', 'nt', 'c', 'ck', 'st', 'k', 'ss', 'sts', 'rd', 'nd',
  'ry', 'rt', 'w', 'lly', 'tt', 'ch', 'ts', 'ty', 'p', 'ls', 'ld', 'nts', 'x', 'rg', 'sh', 'ly', 'th', 'ff', 'g', 'rn', 'ngs',
  'nn', 'tz', 'sm', 'gh', 'ms', 'z', 'cs', 'ps', 'ds', 'b', 'lt', 'nk', 'nds', 'ys', 'rk', 'ght', 'v', 'cks', 'f', 'ct',
  'rth', 'rry', 'lls', 'ny', 'ws', 'cts', 'wn', 'rds', 'dy', 'bly', 'rts', 'ft', 'hl', 'gy', 'pp', 'rly', 'mp', 'ntly',
  'sch', 'ngly', 'sly', 'ks', 'tch', 'ncy', 'rm', 'gs', 'rty', 'hn', 'fy', 'rst', 'rr', 'ntz', 'bs', 'cy', 'dly', 'tts']

onsets = sorted(sorted(set(onsets)),key=len, reverse=True)

nuclei = sorted(sorted(set(nuclei)),key=len, reverse=True)

coda = sorted(sorted(set(coda)),key=len, reverse=True)

def onc_split(word):
    global onsets, nuclei, coda, used_onsets, used_nuclei, used_coda, unidentified_words
    for i in onsets:
        if word.startswith(i):
            i_less = word.replace(i, '', 1)
            for n in nuclei:
                if i_less.startswith(n):
                    n_less = i_less.replace(n, '')
                    return i + '-' + n + '-' + n_less
            break
    for n in nuclei:
                if word.startswith(n):
                    i = ''
                    n_less = word.replace(n, '')
                    return  n + '-' + n_less
                elif n in word:
                    onset, coda = word.split(n, 1)
                    return '-'.join([onset, n, coda])
    return "the rizzlord" + word

def to_categorical(sequences, length):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(length))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)
        return np.array(cat_sequences)

def main():
    st.write('<style>body {background-color: #fce0ff; justify-content: center; align-items: center; height: 100vh;}</style>', unsafe_allow_html=True)
    st.write("""
        <style>
        div.Widget.row-widget.stRadio > div {
            flex-direction:row;
            justify-content:center;
        }
        </style>
        """, unsafe_allow_html=True)

    st.write('<div style="display: flex; flex-direction: column; align-items: center;">', unsafe_allow_html=True)
    st.write('<h1 style="color: #333; font-size: 30px;">Machine Syllabification, Grapheme Identification, and Phonification</h1>', unsafe_allow_html=True)

    input_data = st.text_input('Enter a word or pseudo-word to be syllabified.', max_chars=45)

    syllabify_button = st.button('Syllabify')

    if not re.match("^[a-zA-Z ]*$", input_data):
        with st.spinner("Please enter only alphabetic characters."):
            time.sleep(0.5)
        st.empty()
    else:
        if syllabify_button:
            prediction = piss.syllabify(input_data)
            st.write(f"<div style='font-size: 24px; margin-top: 20px; margin-bottom: 20px;'>{prediction}</div>", unsafe_allow_html=True)
            nuclei_onsets = '-'.join([onc_split(x).strip('-') for x in prediction.split('-')]).split('-')
            if not any("the rizzlord" in n for n in nuclei_onsets):
                st.write(f"<div style='font-size: 24px; margin-top: 20px; margin-bottom: 20px;'>{'-'.join(nuclei_onsets)}</div>", unsafe_allow_html=True)
                if len(nuclei_onsets) <= 19:
                    phon_prediction = ''.join(shit.ipafy(nuclei_onsets))
                    st.write(f"<div style='font-size: 24px; margin-top: 20px; margin-bottom: 20px;'>{phon_prediction}</div>", unsafe_allow_html=True)
                else:
                    st.write(f"<div style='font-size: 24px; margin-top: 20px; margin-bottom: 20px;'>Too many graphemes to process</div>", unsafe_allow_html=True)
            else:
                st.write(f"<div style='font-size: 24px; margin-top: 20px; margin-bottom: 20px;'>{input_data} is unlikely to be an English pronouncable word.= because of {[n.replace("the rizzlord", '') for n in nuclei_onsets if "the rizzlord" in n}</div>", unsafe_allow_html=True)
    
    st.write('<h1 style="color: #333; font-size: 30px;">Ussification</h1>', unsafe_allow_html=True)
    ussify_input = st.text_input('Enter text to ussify.',max_chars=45)
    ussify_button = st.button('Ussify')
    
    if ussify_button:
        ussified_text = Ussy.ussify(ussify_input)
        if ussified_text:
            st.write(f"<div style='font-size: 24px; margin-top: 20px; margin-bottom: 20px;'>{ussified_text}</div>", unsafe_allow_html=True)
        else:
            st.write(f"<div style='font-size: 24px; margin-top: 20px; margin-bottom: 20px;'>{'cannot ussify'}</div>", unsafe_allow_html=True)


    while not syllabify_button:
        time.sleep(0.1)

    st.write('<style>.streamlit-button{opacity: 1 !important;}</style>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()