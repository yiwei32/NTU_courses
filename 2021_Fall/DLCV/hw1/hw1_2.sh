wget -O p2_FCN8s_model.ckpt https://www.dropbox.com/s/haemnu3fzlvoicz/p2_FCN8s_model.ckpt?dl=1 
python3 p2_FCN8s_test.py ./p2_FCN8s_model.ckpt $1 $2
