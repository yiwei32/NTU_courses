if [ $2 == "mnistm" ];then
    wget -O p3_2_mnistm_extractor_model.pth https://www.dropbox.com/s/3n6fzrcomtct7i9/p3_2_mnistm_extractor_model.pth?dl=1
    wget -O p3_2_mnistm_predictor_model.pth https://www.dropbox.com/s/dq2wtdqic63z2sd/p3_2_mnistm_predictor_model.pth?dl=1

elif [ $2 == "usps" ];then
    wget -O p3_2_usps_extractor_model.pth https://www.dropbox.com/s/i7ifa3ipazx6lkz/p3_2_usps_extractor_model.pth?dl=1
    wget -O p3_2_usps_predictor_model.pth https://www.dropbox.com/s/eu71ij49d4g1hzv/p3_2_usps_predictor_model.pth?dl=1

elif [ $2 == "svhn" ];then
    wget -O p3_2_svhn_extractor_model.pth https://www.dropbox.com/s/42zjwj0dyfqlfbr/p3_2_svhn_extractor_model.pth?dl=1
    wget -O p3_2_svhn_predictor_model.pth https://www.dropbox.com/s/t0dczyt7guuaqhi/p3_2_svhn_predictor_model.pth?dl=1
fi  
python3 p3_2_test.py $1 $2 $3
