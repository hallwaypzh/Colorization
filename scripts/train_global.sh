python train_global.py --name siggraph_small_global --sample_p 1.0 --niter 100 --niter_decay 0 --classification --phase train_small --which_model_netG siggraphglobal

mkdir ./checkpoints/siggraph_class_global
cp ./checkpoints/siggraph_small_global/latest_net_G.pth ./checkpoints/siggraph_class_global/
python train_model.py --name siggraph_class --sample_p 1.0 --niter 15 --niter_decay 0 --classification --load_model --phase train --which_model_netG siggraphglobal

# Train regression model (with color hints)
mkdir ./checkpoints/siggraph_reg_global
cp ./checkpoints/siggraph_class_global/latest_net_G.pth ./checkpoints/siggraph_reg_global/
python train_model.py --name siggraph_reg_global --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 --load_model --phase train --which_model_netG siggraphglobal

# Turn down learning rate to 1e-6
mkdir ./checkpoints/siggraph_reg_global2
cp ./checkpoints/siggraph_reg_global/latest_net_G.pth ./checkpoints/siggraph_reg_global2/
python train_model.py --name siggraph_reg_global2 --sample_p .125 --niter 5 --niter_decay 0 --lr 0.000001 --load_model --phase train --which_model_netG siggraphglobal
