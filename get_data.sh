
mkdir res/ logs/ saved_weights/
mkdir res/fasttext-vectors/

wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
mv roberta.large.tar.gz res/
tar -xzf res/roberta.large.tar.gz -C res/
rm res/roberta.large.tar.gz

pip install awscli

aws s3 sync s3://spallas-wsd-us/wsd-test res/wsd-test/
aws s3 sync s3://spallas-wsd-us/wsd-train res/wsd-train/
aws s3 sync s3://spallas-wsd-us/dictionaries res/dictionaries/
