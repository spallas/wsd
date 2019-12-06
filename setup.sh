
mkdir res/ logs/ saved_weights/
mkdir res/fasttext-vectors/
mkdir res/dictionaries/

wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
mv roberta.large.tar.gz res/
tar -xzf res/roberta.large.tar.gz -C res/
rm res/roberta.large.tar.gz

pip install -r requirements.txt --ignore-installed

# [optional]
# git clone https://github.com/NVIDIA/apex
# cd apex || exit
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# cd ..

python -c "import nltk; nltk.download('wordnet')"
