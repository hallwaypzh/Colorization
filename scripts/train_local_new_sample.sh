#mkdir ./checkpoints/siggraph_reg_local_is
#cp ./checkpoints/siggraph_class_local/latest_net_G.pth ./checkpoints/siggraph_reg_local_is/
python train_model.py --name siggraph_reg_local_is --sample_p .125 --niter 4 --niter_decay 0 --lr 0.00001 --load_model --phase train --which_model_netG siggraphlocal --samp importance

#mkdir ./checkpoints/siggraph_reg_local_is2
#cp ./checkpoints/siggraph_reg_local_is/latest_net_G.pth ./checkpoints/siggraph_reg_local_is2/
#python train_model.py --name siggraph_reg_local_is2 --sample_p .125 --niter 3 --niter_decay 0 --lr 0.000001 --load_model --phase train --which_model_netG siggraphlocal --samp importance