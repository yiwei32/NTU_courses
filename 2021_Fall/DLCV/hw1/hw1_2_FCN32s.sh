wget -O p2_FCN32s_model.ckpt https://www.dropbox.com/s/a6e1w574yrpdgoz/p2_FCN32s_model.ckpt?dl=1
python3 p2_FCN32s_test.py ./p2_FCN32s_model.ckpt $1 $2
