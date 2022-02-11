wget -O checkpoint.pth https://www.dropbox.com/s/k46dq54blzl2rpz/checkpoint.pth?dl=1
python3 p2_vis.py --input_path $1 --output_path $2 --checkpoint ./checkpoint.pth