
pip uninstall tokenizers -y
rm -r tokenizers

git clone https://github.com/ecchochan/tokenizers.git
cd tokenizers
git checkout zh-norm
git config user.name "well"
git config user.email "why@here.needtoset"
git merge origin/metaspace-no-consecutive-spaces -m "merge!"

cd bindings/python

# python setup.py install
/opt/conda/bin/python setup.py install

cd ../../..

sudo rm -r tokenizers