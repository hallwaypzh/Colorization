python train_model.py --name siggraph_small_local --sample_p 1.0 --niter 100 --niter_decay 0 --classification --phase train_small --which_model_netG siggraphlocal


mkdir ./checkpoints/siggraph_class_local
cp ./checkpoints/siggraph_small_local/latest_net_G.pth ./checkpoints/siggraph_class_local/
python train_model.py --name siggraph_class_local --sample_p 1.0 --niter 15 --niter_decay 0 --classification --load_model --phase train --which_model_netG siggraphlocal

mkdir ./checkpoints/siggraph_reg_local
cp ./checkpoints/siggraph_class_local/latest_net_G.pth ./checkpoints/siggraph_reg_local/
python train_model.py --name siggraph_reg_local --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 --load_model --phase train --which_model_netG siggraphlocal

# Turn down learning rate to 1e-6
mkdir ./checkpoints/siggraph_reg_local2
cp ./checkpoints/siggraph_reg_local/latest_net_G.pth ./checkpoints/siggraph_reg_local2/
python train_model.py --name siggraph_reg_local2 --sample_p .125 --niter 5 --niter_decay 0 --lr 0.000001 --load_model --phase train --which_model_netG siggraphlocal