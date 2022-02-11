wget -O p1_model.ckpt https://www.dropbox.com/s/8aorjzbsdjmx9e7/model.ckpt?dl=1
python3 test.py ./p1_model.ckpt $1 $2
