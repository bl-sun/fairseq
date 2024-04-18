#!/bin/bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt
BPE_TOKENS=10000

URLS=(
    "/home/sbl/try-huggingface/datasets/mon-120w.txt"
    "/home/sbl/try-huggingface/datasets/zh-120w.txt"
)
FILES=(
    "7w.mn-zh.mn"
    "7w.mn-zh.zh"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=mn
tgt=zh
lang=mn-zh
prep=120w.tokenized.mn-zh

tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig
mkdir -p $lang
cd $lang

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping copy"
    else
        url=${URLS[i]}
        echo "Copy $url to $file"
        cp "$url" "$file"
    fi
done

cd ../..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=7w.$lang.$l
    tok=train.tags.$lang.tok.$l

    # cat $orig/$lang/$f | \
    # grep -v '<url>' | \
    # grep -v '<talkid>' | \
    # grep -v '<keywords>' | \
    # sed -e 's/<title>//g' | \
    # sed -e 's/<\/title>//g' | \
    # sed -e 's/<description>//g' | \
    # sed -e 's/<\/description>//g' | \
    # perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    cat $orig/$lang/$f | \
    perl $TOKENIZER -threads 8 > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

# echo "pre-processing valid/test data..."
# for l in $src $tgt; do
#     for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
#     fname=${o##*/}
#     f=$tmp/${fname%.*}
#     echo $o $f
#     grep '<seg id' $o | \
#         sed -e 's/<seg id="[0-9]*">\s*//g' | \
#         sed -e 's/\s*<\/seg>\s*//g' | \
#         sed -e "s/\’/\'/g" | \
#     perl $TOKENIZER -threads 8 -l $l | \
#     perl $LC > $f
#     echo ""
#     done
# done


echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23-1 == 0)  print $0; }' $tmp/train.tags.$lang.$l > $tmp/test.$l
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$lang.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$lang.$l > $tmp/train.$l

    # cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
    #     $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
    #     $tmp/IWSLT14.TED.tst2010.de-en.$l \
    #     $tmp/IWSLT14.TED.tst2011.de-en.$l \
    #     $tmp/IWSLT14.TED.tst2012.de-en.$l \
    #     > $tmp/test.$l
done

export LC_ALL=C

TRAIN=$tmp/train.en-de
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
