import sys
from pytorch_pretrained_bert import GPT2Tokenizer
import regex as re

pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
enc = GPT2Tokenizer.from_pretrained('gpt2')

filename = sys.argv[1]

with_tldr = False
replace_newline = False
tok_trunc = 1000000

write_name = file_prefix+filename+'.bpe'
if with_tldr and 'src' in filename:
    write_name += '.tldr'

with open(file_prefix+filename, 'r') as f:
    with open(write_name, 'w') as fw:
        for line in f:
            txt = line.strip()
            if with_tldr and 'src' in filename:
                txt += '\nTL;DR:'

            if replace_newline:
                txt = txt.replace('<newline>', '\n')

            bpe_tokens = []
            for token in re.findall(pat, txt): # line.strip() to make sure newline is not encoded
                token = ''.join(enc.byte_encoder[b] for b in token.encode('utf-8'))
                bpe_tokens.extend(enc.bpe(token).split(' '))
            fw.write(' '.join(bpe_tokens[:tok_trunc]) + '\n')
