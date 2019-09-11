import sys
from pytorch_pretrained_bert import GPT2Tokenizer
import regex as re
import argparse

parser = argparse.ArgumentParser(description='Encode text')
parser.add_argument('--input_file', type=str, help='input file')
parser.add_argument('--output_file', type=str, help='full output filename (usually ends in .bpe)')
parser.add_argument('--add_tldr', action='store_true', help='adds \nTL;DR')
parser.add_argument('--replace_newline', action='store_true', help='replace <newline> with \\n')
parser.add_argument('--tok_trunc', type=int, default=1000000, help='truncate tokens')
args = parser.parse_args()

pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
enc = GPT2Tokenizer.from_pretrained('gpt2')

with open(args.input_file, 'r') as f:
    with open(args.output_file, 'w') as fw:
        for line in f:
            txt = line.strip()

            if args.add_tldr:
                txt += '\nTL;DR:'

            if args.replace_newline:
                txt = txt.replace('<newline>', '\n')

            bpe_tokens = []
            for token in re.findall(pat, txt): # line.strip() to make sure newline is not encoded
                token = ''.join(enc.byte_encoder[b] for b in token.encode('utf-8'))
                bpe_tokens.extend(enc.bpe(token).split(' '))
            fw.write(' '.join(bpe_tokens[:args.tok_trunc]) + '\n')
