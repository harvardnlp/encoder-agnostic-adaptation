import argparse

from pytorch_pretrained_bert import GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--src', '-src', type=str)
parser.add_argument('--dst', '-dst', type=str)

args = parser.parse_args()
enc = GPT2Tokenizer.from_pretrained('gpt2')

if args.dst is None:
    if args.src[-4:] == '.bpe':
        args.dst = args.src[:-4]
    elif args.src[-8:] == '.encoded':
        args.dst = args.src[:-8]
    else:
        raise ValueError('dst needed or src that ends in .bpe or .encoded')

i = 0
with open(args.dst, 'w') as fw:
    with open(args.src, 'r') as f:
        for line in f:
            i += 1
            text = line.strip()

            text = ''.join(text.split(' '))

            decoded = bytearray([enc.byte_decoder[c] for c in text]).decode('utf-8', errors=enc.errors)
            decoded = decoded.replace('\n', '') # We need one example per line
            decoded = decoded.replace('\r', '')
            decoded += '\n'
            fw.write(decoded)
print(i)
